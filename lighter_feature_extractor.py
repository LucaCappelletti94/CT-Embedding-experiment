"""Script to extract features from CT scans using a pre-trained model and compute cosine similarity matrices."""

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


def extract_embedding(model, preprocess, input_path):
    """Extract embedding for a single CT scan file."""
    input_tensor = preprocess(input_path)
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))[-1]
        avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()

    return np.array(avg_output)


def compute_embeddings(model, preprocess, file_pattern="*.nii"):
    """Compute embeddings for all CT scan files matching the pattern."""
    return {
        input_path: extract_embedding(model, preprocess, input_path)
        for input_path in tqdm(glob(file_pattern), desc="Embedding CT Scans")
    }


def save_embeddings_to_csv(embeddings, filename="ct_scan_backup.csv"):
    """Save embeddings to a CSV file after scaling."""
    df = pd.DataFrame(embeddings).T
    scaler = RobustScaler().fit(df)
    data = scaler.transform(df)
    df = pd.DataFrame(data, index=df.index, columns=df.columns)
    df.to_csv(filename, index=True)
    return df


def compute_similarity_matrix(df):
    """Compute cosine similarity matrix from embeddings DataFrame."""
    return pd.DataFrame(cosine_similarity(df), columns=df.index, index=df.index)


def main():
    print(f"Torch cuDNN version: {torch.backends.cudnn.version()}")
    preprocess = get_preprocessing_pipeline()
    model = load_model()
    embeddings = compute_embeddings(model, preprocess)
    df = save_embeddings_to_csv(embeddings)
    similarity_matrix = compute_similarity_matrix(df)
    similarity_matrix.to_csv("scores/lighter_feature_extractor.csv")


if __name__ == "__main__":
    main()
