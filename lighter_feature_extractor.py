"""Script to extract features from CT scans using a pre-trained model and compute cosine similarity matrices."""

import os
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import torch
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureType,
    Orientation,
    ScaleIntensityRange,
    CropForeground,
)
from glob import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from cache_decorator import Cache


def get_preprocessing_pipeline():
    """Return the preprocessing pipeline for CT scans."""
    return Compose(
        [
            LoadImage(ensure_channel_first=True),
            EnsureType(),
            Orientation(axcodes="SPL"),
            ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
            CropForeground(),
        ]
    )


def load_model():
    """Load and return the pre-trained SegResEncoder model."""
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model.eval()
    return model


@Cache(
    "{cache_dir}/{function_name}/{input_path}.npy",
    args_to_ignore=["model", "preprocess"],
)
def extract_embedding(model, preprocess, input_path: str) -> np.ndarray:
    """Extract embedding for a single CT scan file."""
    input_tensor = preprocess(input_path)
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))[-1]
        avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()

    return np.array(avg_output)


def compute_embeddings(
    model, preprocess, file_pattern: str = "images/**/*.nii"
) -> pd.DataFrame:
    """Compute embeddings for all CT scan files matching the pattern."""
    return pd.DataFrame(
        {
            input_path: extract_embedding(model, preprocess, input_path)
            for input_path in tqdm(
                glob(file_pattern), desc="Embedding CT Scans", leave=False
            )
        }
    ).T


def normalize_embedding_label(label: str) -> str:
    """Normalize the embedding label by removing specific suffixes."""
    file_name: str = label.split(os.sep)[-1]
    file_name_without_extension: str = file_name.split(".nii")[0]
    number: int = int(file_name_without_extension.split("_")[2])  # Remove leading zeros
    return f"nii.{number}"


def normalize_embeddings(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Save embeddings to a CSV file after scaling."""
    scaler = RobustScaler().fit(embeddings)
    data = scaler.transform(embeddings)
    embeddings = pd.DataFrame(data, index=embeddings.index, columns=embeddings.columns)
    embeddings.index = embeddings.index.map(normalize_embedding_label)
    return embeddings


def compute_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity matrix from embeddings DataFrame."""
    return pd.DataFrame(cosine_similarity(df), columns=df.index, index=df.index)


def main() -> None:
    preprocess = get_preprocessing_pipeline()
    model = load_model()
    approaches = {
        "cropped": "*_cropped.nii",
        "masked_resized": "*_masked_resized.nii",
        "masked": "*_masked.nii",
        "original": "*_[0-9][0-9][0-9].nii",
    }
    for approach_name, pattern in tqdm(list(approaches.items()), desc="Approaches"):
        embeddings = compute_embeddings(
            model, preprocess, file_pattern=f"images/**/{pattern}"
        )
        df = normalize_embeddings(embeddings)
        similarity_matrix = compute_similarity_matrix(df)
        similarity_matrix.to_csv(
            f"scores/lighter_feature_extractor_{approach_name}.csv"
        )


if __name__ == "__main__":
    main()
