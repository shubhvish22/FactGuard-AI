
import os
import sys
import argparse
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TFIDF_MODEL_PATH, LOGREG_MODEL_PATH, MODELS_DIR, DATA_DIR, TEXT_DATASET_DIR
from utils import preprocess_text

LEGACY_DATASET_URL = "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip"
LEGACY_DATASET_PATH = os.path.join(DATA_DIR, "fake_or_real_news.csv")


def load_fakenewsnet_dataset(dataset_dir: str = TEXT_DATASET_DIR, balance: bool = False) -> pd.DataFrame:
    """
    Loads Politifact and Gossipcop claims from backend/dataset.
    Returns clean DataFrame with columns: ['clean_text', 'label'].
    """
    files = [
        ('gossipcop_fake.csv', 'FAKE'),
        ('gossipcop_real.csv', 'REAL'),
        ('politifact_fake.csv', 'FAKE'),
        ('politifact_real.csv', 'REAL'),
    ]
    dfs = []
    print(f"       Loading FakeNewsNet corpus from {dataset_dir}...")
    for fname, label in files:
        p = os.path.join(dataset_dir, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df.dropna(subset=['title'])
            df['clean_text'] = df['title'].astype(str).apply(preprocess_text)
            df['label'] = label
            dfs.append(df[['clean_text', 'label']])
            print(f"         - {fname}: {len(df)} samples ({label})")

    if not dfs:
        raise FileNotFoundError(f"No FakeNewsNet CSV files found in {dataset_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['clean_text'].str.strip().str.len() > 3]
    combined = combined.drop_duplicates(subset=['clean_text'])

    if balance:
        fakes = combined[combined['label'] == 'FAKE']
        reals = combined[combined['label'] == 'REAL']
        n = min(len(fakes), len(reals))
        reals = reals.sample(n=n, random_state=42)
        combined = pd.concat([fakes, reals], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"       Class balance applied: {n} FAKE and {n} REAL samples")

    return combined


def load_legacy_dataset(dataset_path: str = LEGACY_DATASET_PATH) -> pd.DataFrame:
    """Loads article dataset from backend/data/fake_or_real_news.csv."""
    if not os.path.exists(dataset_path):
        print(f"       Legacy dataset not found at {dataset_path}. Downloading...")
        os.makedirs(DATA_DIR, exist_ok=True)
        zip_path = dataset_path + ".zip"
        try:
            urllib.request.urlretrieve(LEGACY_DATASET_URL, zip_path)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
        except Exception as e:
            print(f"       Download failed: {e}")
            return pd.DataFrame(columns=['clean_text', 'label'])

    df = pd.read_csv(dataset_path)
    text_col = "text" if "text" in df.columns else df.columns[1]
    title_col = "title" if "title" in df.columns else None
    label_col = "label" if "label" in df.columns else df.columns[-1]

    df = df.dropna(subset=[text_col, label_col])
    if title_col:
        raw_combined = df[title_col].fillna("").astype(str) + " " + df[text_col].astype(str)
    else:
        raw_combined = df[text_col].astype(str)

    df['clean_text'] = raw_combined.apply(preprocess_text)
    df['label'] = df[label_col].astype(str).str.upper().str.strip()
    df['label'] = df['label'].map(lambda x: 'REAL' if x in ['REAL', '1', 'TRUE', 'RELIABLE'] else 'FAKE')
    print(f"       Loaded legacy article corpus: {len(df)} samples")
    return df[['clean_text', 'label']]


def train(dataset: str = "dataset", balance: bool = False, max_features: int = 35000):
    """
    Train calibrated TF-IDF + Logistic Regression text verification model.
    dataset options:
      - 'dataset' (default): Trains on backend/dataset (FakeNewsNet Politifact & Gossipcop claims).
      - 'combined': Combines backend/dataset with backend/data/fake_or_real_news.csv.
      - 'legacy': Trains on backend/data/fake_or_real_news.csv only.
    """
    print("=" * 70)
    print(f"  FactGuard AI — Text Verification Model Training Pipeline [{dataset.upper()}]")
    print("=" * 70)

    # Step 1: Ingest Data
    print("\n[1/5] Ingesting and preprocessing text corpora...")
    if dataset == "dataset":
        df = load_fakenewsnet_dataset(balance=balance)
    elif dataset == "combined":
        df_fnn = load_fakenewsnet_dataset(balance=balance)
        df_leg = load_legacy_dataset()
        df = pd.concat([df_fnn, df_leg], ignore_index=True)
        df = df[df['clean_text'].str.strip().str.len() > 3].drop_duplicates(subset=['clean_text'])
    elif dataset == "legacy":
        df = load_legacy_dataset()
    else:
        raise ValueError(f"Unknown dataset option: {dataset}")

    print(f"       Total training pool: {len(df)} samples")
    print(f"       Class distribution:\n{df['label'].value_counts().to_string(header=False)}")

    X = df['clean_text']
    y = df['label']
    y_bin = (y == 'FAKE').astype(int)

    # Step 2: Strict Featurization Ordering
    print("\n[2/5] Strict featurization split (80% train / 20% test)...")
    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y, y_bin, test_size=0.20, random_state=42, stratify=y
    )
    print(f"       Training set: {len(X_train)} | Holdout test set: {len(X_test)}")

    # Step 3: Feature Extraction (Fitted on Training Data ONLY)
    print(f"\n[3/5] Extracting TF-IDF n-grams (max_features={max_features}, ngrams=1-2)...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=2,
        max_df=0.90,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"       TF-IDF vocabulary built: {X_train_tfidf.shape[1]} features")

    # Step 4: Calibrated Classifier Training
    print("\n[4/5] Training Calibrated Logistic Regression model (5-fold cross-validation)...")
    base_model = LogisticRegression(
        max_iter=1000,
        C=2.5,
        solver="lbfgs",
        class_weight="balanced" if not balance else None,
        random_state=42,
    )
    calibrated_model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
    calibrated_model.fit(X_train_tfidf, y_train)

    # Step 5: Model Evaluation on Holdout Test Split
    print("\n[5/5] Evaluating calibrated model on holdout test split...")
    y_pred = calibrated_model.predict(X_test_tfidf)
    fake_idx = list(calibrated_model.classes_).index('FAKE')
    y_proba = calibrated_model.predict_proba(X_test_tfidf)[:, fake_idx]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(yb_test, y_proba)

    print("\n" + "=" * 70)
    print(f"  HOLDOUT EVALUATION REPORT ({dataset.upper()} - 20% Stratified Split)")
    print("=" * 70)
    print(f"  Holdout Accuracy : {acc * 100:.2f}%")
    print(f"  ROC-AUC Score    : {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    # Save Model Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_MODEL_PATH)
    joblib.dump(calibrated_model, LOGREG_MODEL_PATH)
    print(f"\n[SAVED] TF-IDF Vectorizer -> {TFIDF_MODEL_PATH}")
    print(f"[SAVED] Calibrated Model   -> {LOGREG_MODEL_PATH}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FactGuard AI — Text Verification Model Trainer")
    parser.add_argument("--dataset", choices=["dataset", "combined", "legacy"], default="dataset",
                        help="Corpus to train on: 'dataset' (backend/dataset FakeNewsNet), 'combined', or 'legacy'")
    parser.add_argument("--balance", action="store_true",
                        help="Balance positive and negative classes evenly before training")
    parser.add_argument("--max-features", type=int, default=35000,
                        help="Maximum TF-IDF vocabulary size (default: 35000)")
    args = parser.parse_args()

    train(dataset=args.dataset, balance=args.balance, max_features=args.max_features)
