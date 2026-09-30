
import torch
import torch.nn.functional as F
from torch import nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) with sigmoid gating."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as specified in TFT architecture."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            # Broadcast context across sequence if needed
            if context.dim() == 2 and h.dim() == 3:
                c = self.context_fc(context).unsqueeze(1)
            else:
                c = self.context_fc(context)
            h = h + c
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        gated = self.glu(h)
        return self.layer_norm(self.skip(x) + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for weighting individual feature contributions."""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.var_transforms = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.gating = GatedResidualNetwork(
            input_dim=num_features * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, num_features) or (batch, seq_len, num_features)
        transformed = []
        for i in range(self.num_features):
            feat = x[..., i:i+1]
            transformed.append(self.var_transforms[i](feat))
        # transformed: list of tensors of shape (..., hidden_dim)
        stacked = torch.stack(transformed, dim=-2)  # (..., num_features, hidden_dim)
        
        # Flatten for gating network
        flat = stacked.flatten(start_dim=-2)
        sparse_weights = F.softmax(self.gating(flat, context), dim=-1)  # (..., num_features)
        
        # Weighted sum across features
        weights_expanded = sparse_weights.unsqueeze(-1)  # (..., num_features, 1)
        selected = (stacked * weights_expanded).sum(dim=-2)  # (..., hidden_dim)
        return selected, sparse_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention mechanism."""
    def __init__(self, hidden_dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = q.shape
        Q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        return output, attn_weights.mean(dim=1)


class VesselPerformanceTFT(nn.Module):
    """
    Temporal Fusion Transformer (TFT) specialized for maritime vessel performance.
    
    Predicts expected future speed over ground (SOG in knots) given:
    1. Historical AIS sequence: SOG, COG (sin/cos), Heading (sin/cos), step displacement.
    2. Static Vessel Context: Ship type embedding, design speed, dimensions.
    3. Dynamic Environmental Context: Wind speed, wind relative direction, wave height,
       wave direction, wave period, swell height, sea surface temperature, ocean currents.
    4. Route Context: Segment bearing, distance.
    """
    def __init__(
        self,
        hist_num_features: int = 6,
        env_num_features: int = 10,
        num_ship_types: int = 16,
        ship_embed_dim: int = 8,
        hidden_dim: int = 48,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Vessel Static Embedding & GRN
        self.ship_type_embed = nn.Embedding(num_ship_types, ship_embed_dim)
        self.static_vessel_fc = nn.Linear(ship_embed_dim + 4, hidden_dim)  # +4 for design_speed, length, beam, draft
        self.static_context_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # 2. Historical Sequence VSN + LSTM
        self.hist_vsn = VariableSelectionNetwork(hist_num_features, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.lstm_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 3. Self-Attention over Historical Sequence
        self.temporal_attn = InterpretableMultiHeadAttention(hidden_dim, num_heads=num_heads)
        self.attn_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 4. Environmental & Route Context VSN
        self.env_vsn = VariableSelectionNetwork(env_num_features, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.env_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 5. Fusion & Output Head
        self.fusion_grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        hist_seq: torch.Tensor,       # (batch, seq_len=30, hist_num_features)
        ship_type_id: torch.Tensor,   # (batch,)
        vessel_stats: torch.Tensor,   # (batch, 4): normalized [design_speed, length, beam, draft]
        env_features: torch.Tensor,   # (batch, env_num_features)
    ) -> dict[str, torch.Tensor]:
        # Step 1: Compute Static Context Representation
        type_emb = self.ship_type_embed(ship_type_id)
        vessel_raw = torch.cat([type_emb, vessel_stats], dim=-1)
        vessel_emb = F.relu(self.static_vessel_fc(vessel_raw))
        static_context = self.static_context_grn(vessel_emb)  # (batch, hidden_dim)

        # Step 2: Temporal Historical Processing
        hist_selected, hist_weights = self.hist_vsn(hist_seq, context=static_context)
        lstm_out, _ = self.lstm(hist_selected)
        lstm_post = self.lstm_post_grn(lstm_out, context=static_context)

        # Step 3: Self-Attention over Historical Time Steps
        attn_out, attn_weights = self.temporal_attn(lstm_post, lstm_post, lstm_post)
        attn_post = self.attn_post_grn(attn_out + lstm_post, context=static_context)
        # Pool latest temporal state
        hist_summary = attn_post[:, -1, :]  # (batch, hidden_dim)

        # Step 4: Environmental Context Processing
        env_selected, env_weights = self.env_vsn(env_features, context=static_context)
        env_post = self.env_post_grn(env_selected, context=static_context)  # (batch, hidden_dim)

        # Step 5: Fusion of Trajectory State and Sea/Route Conditions
        fused_repr = torch.cat([hist_summary, env_post], dim=-1)  # (batch, hidden_dim * 2)
        fused = self.fusion_grn(fused_repr, context=static_context)

        # Step 6: Prediction Head (speed in knots >= 0)
        raw_pred = self.head(fused).squeeze(-1)
        pred_speed = F.softplus(raw_pred)  # Non-negative predicted SOG

        return {
            "predicted_sog_kn": pred_speed,
            "temporal_weights": attn_weights,
            "hist_feature_weights": hist_weights,
            "env_feature_weights": env_weights,
        }
