import os
import shutil
import pandas as pd
import re
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_filtered_yolo_dataset(
    filtered_csv: str = "data/training/opensar/yolo_dataset_filtered.csv",
    source_dir: str = "data/processed/opensar_yolo",
    target_dir: str = "data/processed/opensar_filtered",
):
    df = pd.read_csv(filtered_csv)

    def to_base_id(fn):
        m = re.search(r'(DB_OPENSAR_)(?:VV|VH)_(VD_\d+)', fn)
        return f"{m.group(1)}{m.group(2)}" if m else fn

    df["base_id"] = df["file_name"].apply(to_base_id)

    os.makedirs(target_dir, exist_ok=True)
    stats = {}

    for split in ["train", "val", "test"]:
        img_out = os.path.join(target_dir, "images", split)
        lbl_out = os.path.join(target_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        split_df = df[df["split"] == split]
        unique_bases = split_df["base_id"].unique()

        copied = 0
        total_vessels = 0
        for base_id in unique_bases:
            src_img = os.path.join(source_dir, "images", split, f"{base_id}.jpg")
            src_lbl = os.path.join(source_dir, "labels", split, f"{base_id}.txt")

            if os.path.exists(src_img) and os.path.exists(src_lbl):
                dst_img = os.path.join(img_out, f"{base_id}.jpg")
                dst_lbl = os.path.join(lbl_out, f"{base_id}.txt")

                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)

                with open(src_lbl) as f:
                    lines = [l.strip() for l in f if l.strip()]
                    total_vessels += len(lines)
                copied += 1

        stats[split] = {"images": copied, "vessels": total_vessels}

    dataset_yaml = {
        "path": os.path.abspath(target_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [1, 3],
        "names": {
            0: "vessel",
        },
    }

    yaml_path = os.path.join(target_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    logger.info("Filtered dataset created at %s", target_dir)
    logger.info("Dataset statistics: %s", stats)


if __name__ == "__main__":
    build_filtered_yolo_dataset()
