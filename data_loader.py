"""
Data loading module for the Fake Reviews Dataset (Salminen et al., 2020)
Fallback sources: Kaggle → OSF → HuggingFace → Synthetic Data Generator
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "fake_reviews_dataset.csv"

def generate_synthetic_data(n_genuine=1000, n_fake=1000):
    """
    Generate synthetic but realistic fake vs genuine review pairs for demo.
    """
    print("Generating synthetic dataset for demo...")
    np.random.seed(42)

    genuine_reviews = [
        "Great product! Arrived quickly and works perfectly.",
        "Very satisfied with this purchase. Quality is excellent.",
        "Exceeded my expectations. Highly recommend to others.",
        "Good value for money. Exactly as described.",
        "Love it! Best purchase I've made this year.",
        "Product works as intended. Very happy with it.",
        "Quality is amazing. Shipping was fast too.",
        "Exactly what I needed. Great service!",
        "Fantastic product. Will buy again for sure.",
        "Amazing quality at this price point.",
        "So happy I found this. Perfect!",
        "Incredible value. Does what it says.",
        "Best investment ever made.",
        "Really impressed with the quality.",
        "Can't believe how good this is!",
        "Absolutely love this product!",
        "Perfect fit and quality.",
        "Exceeded all my expectations!",
        "Worth every penny.",
        "Top notch service and product!",
    ]

    fake_reviews = [
        "This product is amazing!!!!! Five stars!!!!",
        "Best thing I ever bought!!! Highly recommend!!!!!!",
        "Amazing quality!!! Love it!!! Will buy again!!!",
        "Perfect product!!! Everyone should have this!!!",
        "This changed my life!!! Incredible!!!",
        "Simply the best!!! No complaints!!!",
        "Cannot express how great this is!!!",
        "5 stars!!! Highly recommended!!!",
        "This is what everyone is buying!!!",
        "Professional grade quality!!!",
        "Must have product!!!",
        "Game changer!!!",
        "Industry leading!!!",
        "Outstanding performance!!!",
        "Unbelievable value!!!",
        "Cannot recommend enough!!!",
        "This is incredible!!!",
        "Best purchase ever made!!!",
        "Would give 10 stars!!!",
        "Premium quality!!!",
    ]

    reviews = []
    labels = []

    # Generate genuine reviews with variations
    for i in range(n_genuine):
        base_review = np.random.choice(genuine_reviews)
        # Add some natural variations
        if np.random.random() > 0.7:
            base_review += f" I have been using this for {np.random.randint(1, 12)} months now."
        if np.random.random() > 0.6:
            base_review += f" The {np.random.choice(['color', 'size', 'design', 'quality'])} is perfect."
        reviews.append(base_review)
        labels.append(0)  # Genuine

    # Generate fake reviews with more templates and exclamation marks
    for i in range(n_fake):
        base_review = np.random.choice(fake_reviews)
        # Fake reviews tend to repeat exclamation marks
        if np.random.random() > 0.3:
            base_review += "!!!!"
        if np.random.random() > 0.5:
            base_review += f" Rating: {np.random.randint(4, 6)}/5 stars."
        reviews.append(base_review)
        labels.append(1)  # Fake

    df = pd.DataFrame({
        'review_text': reviews,
        'label': labels
    })

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df

def download_from_kaggle():
    """Download from Kaggle using kagglehub API"""
    try:
        import kagglehub
        print("Attempting download from Kaggle (mexwell/fake-reviews-dataset)...")
        path = kagglehub.dataset_download("mexwell/fake-reviews-dataset")

        # Find the CSV file in the downloaded directory
        for file in Path(path).glob("*.csv"):
            if "review" in file.name.lower():
                df = pd.read_csv(file)
                return df

        # If no obvious review CSV, try the first one
        csvs = list(Path(path).glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0])
            return df

    except Exception as e:
        print(f"Kaggle download failed: {e}")
    return None

def download_from_osf():
    """Download from Open Science Framework (OSF) repository"""
    try:
        print("Attempting download from OSF repository...")
        # Alternative OSF endpoint
        osf_url = "https://files.osf.io/v1/resources/4pxsr/providers/googledrive/?kind=file"
        response = requests.get(osf_url, timeout=10, stream=False)

        if response.status_code == 200:
            # Parse response to find CSV download link
            data = response.json()
            # This might be complex, skip for now

        # Try direct download URL pattern
        osf_direct = "https://osf.io/download/mryx5/"
        response = requests.get(osf_direct, timeout=30, allow_redirects=True)
        response.raise_for_status()

        zip_path = DATA_DIR / "reviews.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # Find CSV files
        for file in DATA_DIR.glob("**/*.csv"):
            if "review" in file.name.lower() or file.name.endswith(".csv"):
                df = pd.read_csv(file)
                if len(df) > 0:
                    return df

    except Exception as e:
        print(f"OSF download failed: {e}")
    return None

def download_from_huggingface():
    """Download from HuggingFace Hub"""
    try:
        print("Attempting download from HuggingFace Hub...")
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id="datasets/mexwell/fake-reviews-dataset",
            filename="fake_reviews.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
    return None

def load_dataset(force_resample=False, sample_size=5000):
    """
    Load fake reviews dataset with fallback sources.
    Falls back to synthetic data if all external sources fail.

    Args:
        force_resample: if True and dataset > sample_size, stratified subsample
        sample_size: target sample size if resampling

    Returns:
        df: pandas DataFrame with reviews
        sample_note: string explaining if/how dataset was sampled
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Try to load from local cache first
    if DATASET_PATH.exists():
        print(f"Loading dataset from cache: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        sample_note = ""
    else:
        # Try Kaggle first
        df = download_from_kaggle()

        # Then OSF
        if df is None:
            df = download_from_osf()

        # Then HuggingFace
        if df is None:
            df = download_from_huggingface()

        # Fall back to synthetic data
        if df is None:
            print("All external sources unavailable. Generating synthetic dataset for demo...")
            df = generate_synthetic_data(n_genuine=1000, n_fake=1000)
            sample_note = " (synthetic demo data)"
        else:
            print(f"Downloaded dataset: {len(df)} rows")
            sample_note = ""

        df.to_csv(DATASET_PATH, index=False)

    # Clean dataset
    original_size = len(df)

    # Normalize column names (handle various spellings)
    df.columns = df.columns.str.lower().str.strip()

    # Try to find the text column (must do this before deduping)
    text_col = None
    for col in ['text_', 'review_text', 'text', 'review', 'content']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Could not find review text column. Available columns: {list(df.columns)}"
        )

    # Remove duplicates based on text column
    df = df.drop_duplicates(subset=[text_col], keep='first')

    # Try to find the label column
    label_col = None
    for col in ['label', 'category', 'class', 'cg_or_or']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            f"Could not find label column. Available columns: {list(df.columns)}"
        )

    # Keep only relevant columns
    df = df[[text_col, label_col]].copy()
    df.columns = ['review_text', 'label']
    df = df.dropna(subset=['review_text', 'label'])

    # Make sure text is string and remove any null/nan text
    df['review_text'] = df['review_text'].astype(str)
    df = df[df['review_text'].str.strip() != '']
    df = df[df['review_text'] != 'nan']

    # Normalize labels: OR/Genuine -> 0, CG/Computer Generated/Fake -> 1
    label_map = {
        'OR': 0, 'or': 0, 0: 0,
        'CG': 1, 'cg': 1, 1: 1,
        'Genuine': 0, 'genuine': 0,
        'Computer Generated': 1, 'computer generated': 1, 'Fake': 1, 'fake': 1,
    }

    # Convert to string for mapping
    df['label'] = df['label'].astype(str).str.strip()
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Check if resampling is needed
    if force_resample or len(df) > sample_size * 1.5:
        if len(df) > sample_size:
            print(f"Dataset size ({len(df)} rows) exceeds target. Stratified resampling to {sample_size}...")
            df = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max(1, int(sample_size * len(x) / len(df)))),
                                   random_state=42)
            ).sample(frac=1.0, random_state=42).reset_index(drop=True)
            sample_note = f" (stratified sample of {len(df)} from {original_size} original)"

    print(f"Final dataset: {len(df)} rows, Label distribution: {df['label'].value_counts().to_dict()}{sample_note}")

    return df, sample_note

def split_data(df, train_size=0.8, random_state=42):
    """Split data into train/test with stratification"""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df['label']
    )

    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    return train_df, test_df
