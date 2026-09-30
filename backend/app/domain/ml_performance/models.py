from pydantic import BaseModel, Field


class PerformancePrediction(BaseModel):
    """Result of deep-learning vessel performance inference."""
    calm_reference_speed_kn: float = Field(..., description="Calm-water reference speed in knots")
    predicted_sog_kn: float = Field(..., description="AI-predicted speed over ground in knots under current sea state")
    speed_loss_kn: float = Field(..., description="Predicted speed reduction due to weather exposure")
    speed_retention_ratio: float = Field(..., description="Ratio of predicted speed to calm design speed [0..1]")
    feature_attributions: dict[str, float] | None = None
