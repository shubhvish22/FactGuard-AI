
import os
import sys
import io
import glob
import time
import zipfile
import logging
import concurrent.futures
from typing import List, Tuple, Optional

import requests
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Ensure backend directory is in path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from config import FLORIDA_DIR, DATA_2_DIR, DATA_3_DIR, FLORIDA_MODEL_PATH, MODELS_DIR, DATA_DIR
from verification_engine.image_verifier import _load_hf_image_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FactGuard.ImageTrainer")

CACHE_DIR = os.path.join(DATA_DIR, "florida_cache")
REAL_DIR = os.path.join(CACHE_DIR, "real")
AI_DIR = os.path.join(CACHE_DIR, "ai")
FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "combined_features_v2_14d.npz")
DATA_3_CACHE_PATH = os.path.join(CACHE_DIR, "data3_features_14d.npz")


def extract_florida_dataset(max_images: int = 400) -> List[str]:
    """
    Extracts real photographs from FLORIDA zip archives (PART1-4).
    Filters out OS artifacts and verifies image integrity.
    """
    os.makedirs(REAL_DIR, exist_ok=True)
    existing = [
        os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]
    if len(existing) >= max_images:
        logger.info("Found %d already extracted FLORIDA images in %s", len(existing), REAL_DIR)
        return existing[:max_images]

    logger.info("Extracting FLORIDA dataset archives from %s...", FLORIDA_DIR)
    zip_files = sorted(glob.glob(os.path.join(FLORIDA_DIR, "PART*.zip")))
    if not zip_files:
        logger.warning("No FLORIDA zip archives found in %s; using existing cache if present", FLORIDA_DIR)
        return existing[:max_images]

    extracted_paths = list(existing)
    count = len(extracted_paths)

    for zf_path in zip_files:
        if count >= max_images:
            break
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                for member in zf.namelist():
                    if count >= max_images:
                        break
                    # Skip macOS metadata and non-image files
                    if member.startswith("__MACOSX") or "/._" in member or member.startswith("._"):
                        continue
                    if not member.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        continue

                    data = zf.read(member)
                    try:
                        img = Image.open(io.BytesIO(data))
                        img.verify()
                    except Exception:
                        continue

                    base_name = os.path.basename(member)
                    out_path = os.path.join(REAL_DIR, f"florida_{len(extracted_paths):04d}_{base_name}")
                    with open(out_path, "wb") as f_out:
                        f_out.write(data)

                    extracted_paths.append(out_path)
                    count += 1
        except Exception as e:
            logger.warning("Error reading archive %s: %s", zf_path, e)

    logger.info("Extracted %d valid real images from FLORIDA dataset into %s", len(extracted_paths), REAL_DIR)
    return extracted_paths[:max_images]


def load_data2_samples(
    data_dir: str = DATA_2_DIR,
    split: str = "train",
    max_per_class: int = 400,
) -> Tuple[List[str], List[str]]:
    """
    Loads file paths from the data_2 dataset (CIFAKE).
    Returns (fake_paths, real_paths).
    """
    split_dir = os.path.join(data_dir, split)
    fake_dir = os.path.join(split_dir, "FAKE")
    real_dir = os.path.join(split_dir, "REAL")

    fake_paths = []
    real_paths = []

    if os.path.isdir(fake_dir):
        files = sorted([
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        fake_paths = files[:max_per_class]

    if os.path.isdir(real_dir):
        files = sorted([
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        real_paths = files[:max_per_class]

    logger.info("Loaded %d FAKE and %d REAL samples from data_2/%s", len(fake_paths), len(real_paths), split)
    return fake_paths, real_paths


def load_data3_samples(
    data3_dir: str = DATA_3_DIR,
    max_ai_art: int = 600,
    max_real_art: int = 600,
    max_deepfake_fake: int = 600,
    max_deepfake_real: int = 600,
    include_intel_scenes: bool = True,
    max_intel_scenes: int = 216,
) -> Tuple[List[str], List[str]]:
    """
    Loads Real and Fake/AI/Deepfake image file paths from the data_3 directory:
      1. AI Generated Images vs Real Images:
         - AiArtData (AI / Fake = 1)
         - RealArt (Real = 0)
      2. DeepFake image detection:
         - /fake/ folders (Deepfakes / Manipulated = 1)
         - /real/ folders (Authentic facial portraits = 0)
      3. Intel Image Classification:
         - seg_train / seg_train (Diverse natural scenes: buildings, forest, glacier, mountain, sea, street = 0)
    Returns:
      (fake_image_paths, real_image_paths)
    """
    fake_paths = []
    real_paths = []

    if not os.path.isdir(data3_dir):
        logger.warning("data_3 directory not found at: %s", data3_dir)
        return fake_paths, real_paths

    # 1. AI Generated Images vs Real Images
    ai_art_dir = os.path.join(data3_dir, "AI Generated Images vs Real Images", "AiArtData", "AiArtData")
    real_art_dir = os.path.join(data3_dir, "AI Generated Images vs Real Images", "RealArt", "RealArt")

    if os.path.isdir(ai_art_dir):
        files = sorted([
            os.path.join(ai_art_dir, f) for f in os.listdir(ai_art_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        fake_paths.extend(files[:max_ai_art])

    if os.path.isdir(real_art_dir):
        files = sorted([
            os.path.join(real_art_dir, f) for f in os.listdir(real_art_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        real_paths.extend(files[:max_real_art])

    # 2. DeepFake image detection
    deepfake_base = os.path.join(data3_dir, "DeepFake image detection")
    df_fakes = []
    df_reals = []
    if os.path.isdir(deepfake_base):
        for root, dirs, files in os.walk(deepfake_base):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    p = os.path.join(root, f)
                    norm = p.replace('\\', '/').lower()
                    if '/fake/' in norm or norm.endswith('/fake'):
                        df_fakes.append(p)
                    elif '/real/' in norm or norm.endswith('/real'):
                        df_reals.append(p)
        fake_paths.extend(df_fakes[:max_deepfake_fake])
        real_paths.extend(df_reals[:max_deepfake_real])

    # 3. Intel Image Classification scenes (balanced negative real examples)
    if include_intel_scenes:
        intel_base = os.path.join(data3_dir, "Intel Image Classification", "seg_train", "seg_train")
        if os.path.isdir(intel_base):
            subcats = sorted(os.listdir(intel_base))
            per_sub = max(1, max_intel_scenes // len(subcats)) if subcats else 36
            intel_reals = []
            for sub in subcats:
                sub_p = os.path.join(intel_base, sub)
                if os.path.isdir(sub_p):
                    imgs = [
                        os.path.join(sub_p, f) for f in os.listdir(sub_p)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                    ]
                    intel_reals.extend(imgs[:per_sub])
            real_paths.extend(intel_reals[:max_intel_scenes])

    logger.info("Loaded from data_3: %d FAKE/AI/DeepFake and %d REAL images", len(fake_paths), len(real_paths))
    return fake_paths, real_paths


def fetch_ai_generated_dataset(target_count: int = 400) -> List[str]:
    """
    Retrieves diverse AI-generated images from local cache or remote verified repositories.
    """
    os.makedirs(AI_DIR, exist_ok=True)
    existing = [
        os.path.join(AI_DIR, f) for f in os.listdir(AI_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]
    if len(existing) >= target_count:
        logger.info("Found %d already downloaded AI images in %s", len(existing), AI_DIR)
        return existing[:target_count]

    logger.info("Fetching AI-generated image index from HuggingFace...")
    index_url = "https://huggingface.co/api/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/tree/main/AiArtData/AiArtData"
    try:
        resp = requests.get(index_url, timeout=15)
        resp.raise_for_status()
        items = resp.json()
        file_paths = [item["path"] for item in items if isinstance(item, dict) and "path" in item]
    except Exception as e:
        logger.warning("Could not fetch remote AI index (%s). Using local cache.", e)
        file_paths = []

    base_url = "https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets/resolve/main/"
    download_targets = file_paths[:target_count]
    downloaded_paths = list(existing)

    def download_one(rel_path: str):
        file_name = os.path.basename(rel_path)
        dest_path = os.path.join(AI_DIR, f"ai_{file_name}")
        if os.path.exists(dest_path):
            return dest_path
        try:
            r = requests.get(base_url + rel_path, timeout=12)
            if r.status_code == 200 and len(r.content) > 5000:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return dest_path
        except Exception:
            pass
        return None

    if download_targets:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(download_one, download_targets))
        for path in results:
            if path and path not in downloaded_paths:
                downloaded_paths.append(path)

    logger.info("Total AI-generated training images available: %d", len(downloaded_paths))
    return downloaded_paths[:target_count]


def extract_forensics_only(img: Image.Image) -> np.ndarray:
    """
    Computes 12 resolution-normalized physical, spectral, and forensic features.
    Canonical resolution resizing to 256x256 ensures complete scale and resolution invariance.
    """
    try:
        img_rgb = img.convert("RGB")
        # Canonical reference resolution
        canonical = img_rgb.resize((256, 256), Image.LANCZOS)
        arr = np.array(canonical, dtype=np.float64)

        # 1. Laplacian Camera Sensor Shot Noise Variance
        gray = canonical.convert("L")
        lap = gray.filter(ImageFilter.FIND_EDGES)
        lap_arr = np.array(lap, dtype=np.float64)
        noise_var = float(np.var(lap_arr))

        # 2. Local Patch Texture CV (Coefficient of Variation across 32x32 blocks)
        patch_size = 32
        patches = []
        for y in range(0, 256 - patch_size, patch_size):
            for x in range(0, 256 - patch_size, patch_size):
                p = arr[y:y+patch_size, x:x+patch_size]
                patches.append(np.std(p))
        mean_p = float(np.mean(patches)) if patches else 1.0
        std_p = float(np.std(patches)) if patches else 0.0
        texture_cv = std_p / (mean_p + 1e-5)

        # 3. FFT Multi-band Spectral Energy Ratios (Mid and High Frequency)
        gray_arr = np.array(gray, dtype=np.float64)
        f = np.fft.fft2(gray_arr)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) + 1e-8
        tot_energy = np.sum(mag)
        cy, cx = 128, 128
        y_c, x_c = np.ogrid[:256, :256]
        dist_sq = (x_c - cx)**2 + (y_c - cy)**2

        r_mid = 48**2
        r_high = 80**2
        fft_high = float(np.sum(mag[dist_sq > r_high]) / tot_energy)
        fft_mid = float(np.sum(mag[(dist_sq > r_mid) & (dist_sq <= r_high)]) / tot_energy)

        # 4. FFT Radial Azimuthal Asymmetry (Detects diffusion lattice and transposed convolution spikes)
        q1 = np.sum(mag[:128, :128])
        q2 = np.sum(mag[:128, 128:])
        q3 = np.sum(mag[128:, :128])
        q4 = np.sum(mag[128:, 128:])
        quads = [q1, q2, q3, q4]
        fft_radial_asym = float(np.std(quads) / (np.mean(quads) + 1e-5))

        # 5. Contrast-Normalized Error Level Analysis (ELA)
        buf = io.BytesIO()
        canonical.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        comp = Image.open(buf)
        diff = ImageChops.difference(canonical, comp)
        diff_arr = np.array(diff, dtype=np.float64)
        max_d = float(np.max(diff_arr))
        scale = (255.0 / max_d) if max_d > 0 else 1.0
        diff_scaled = diff_arr * scale
        ela_mean = float(np.mean(diff_scaled))
        ela_var = float(np.var(diff_scaled))

        # 6. Sobel Edge Gradient Density & Edge Coherence
        edges = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edge_arr = np.array(edges, dtype=np.float64)
        edge_density = float(np.mean(edge_arr > 40))
        edge_coherence = float(np.sum(edge_arr > 80) / (np.sum(edge_arr > 20) + 1e-5))

        # 7. Bayer CFA Cross-Channel Gradient Correlation
        r_ch = arr[:, :, 0]
        g_ch = arr[:, :, 1]
        b_ch = arr[:, :, 2]
        grad_r = np.abs(r_ch[:, 1:] - r_ch[:, :-1])
        grad_g = np.abs(g_ch[:, 1:] - g_ch[:, :-1])
        grad_b = np.abs(b_ch[:, 1:] - b_ch[:, :-1])
        std_r, std_g, std_b = np.std(grad_r), np.std(grad_g), np.std(grad_b)
        corr_rg = float(np.corrcoef(grad_r.flatten(), grad_g.flatten())[0, 1]) if std_r > 0 and std_g > 0 else 0.5
        corr_bg = float(np.corrcoef(grad_b.flatten(), grad_g.flatten())[0, 1]) if std_b > 0 and std_g > 0 else 0.5
        cfa_score = float((corr_rg + corr_bg) / 2.0)

        # 8. HSV Color Saturation Entropy
        hsv = canonical.convert("HSV")
        sat = np.array(hsv)[:, :, 1]
        sat_hist, _ = np.histogram(sat, bins=16, density=True)
        sat_hist = sat_hist[sat_hist > 0]
        sat_entropy = float(-np.sum(sat_hist * np.log2(sat_hist))) if len(sat_hist) > 0 else 0.0

        # 9. Luminance Distribution Skewness
        lum = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
        lum_std = float(np.std(lum))
        lum_skew = float(np.mean(((lum - np.mean(lum)) / (lum_std + 1e-5))**3)) if lum_std > 1e-4 else 0.0

        return np.array([
            noise_var,
            texture_cv,
            fft_high,
            fft_mid,
            fft_radial_asym,
            ela_mean,
            ela_var,
            edge_density,
            edge_coherence,
            cfa_score,
            sat_entropy,
            lum_skew,
        ], dtype=np.float64)

    except Exception:
        return np.zeros(12, dtype=np.float64)


def extract_features_from_pil(raw_img: Image.Image, hf_pipeline) -> Optional[np.ndarray]:
    """
    Extracts a 14-dimensional multimodal feature vector from a PIL Image:
      [0]  ViT Artificial Probability
      [1]  ViT Human Probability
      [2]  Laplacian Camera Sensor Noise Variance (Normalized)
      [3]  Local Patch Texture Coefficient of Variation
      [4]  Fourier High-Frequency Energy Ratio
      [5]  Fourier Mid-Frequency Energy Ratio
      [6]  Fourier Radial Azimuthal Asymmetry
      [7]  Contrast-Normalized ELA Mean Discrepancy
      [8]  Contrast-Normalized ELA Variance
      [9]  Sobel Edge Gradient Density
      [10] Edge Coherence (Sharpness vs Diffusion blur)
      [11] Bayer CFA Cross-Channel Gradient Correlation
      [12] Color Saturation Entropy
      [13] Luminance Distribution Skewness
    """
    try:
        img_rgb = raw_img.convert("RGB")
        vit_art = 0.5
        vit_hum = 0.5
        if hf_pipeline is not None:
            try:
                hf_out = hf_pipeline(img_rgb)
                for item in hf_out:
                    lbl = item.get("label", "").lower()
                    sc = float(item.get("score", 0.0))
                    if any(k in lbl for k in ("art", "fake", "ai", "syn")):
                        vit_art = sc
                    elif any(k in lbl for k in ("hum", "real", "auth")):
                        vit_hum = sc
            except Exception:
                pass

        forensics = extract_forensics_only(img_rgb)

        return np.concatenate([np.array([vit_art, vit_hum], dtype=np.float64), forensics])

    except Exception:
        return None


def extract_features_batch(
    image_paths: List[str],
    hf_pipeline,
    batch_size: int = 32,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Fast batched feature extraction for a list of image file paths:
      - Loads PIL images in batches
      - Executes batched ViT tensor inference in PyTorch
      - Multi-threaded forensic feature extraction in Python
    Returns (valid_feature_vectors, valid_indices).
    """
    features_list = []
    valid_indices = []

    total = len(image_paths)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_paths = image_paths[start_idx:end_idx]

        batch_imgs = []
        batch_img_indices = []
        for local_idx, path in enumerate(batch_paths):
            try:
                with Image.open(path) as img:
                    batch_imgs.append(img.convert("RGB"))
                    batch_img_indices.append(start_idx + local_idx)
            except Exception:
                continue

        if not batch_imgs:
            continue

        # 1. Batch ViT inference
        vit_scores = []
        if hf_pipeline is not None:
            try:
                hf_results = hf_pipeline(batch_imgs, batch_size=len(batch_imgs))
                for res in hf_results:
                    v_art = 0.5
                    v_hum = 0.5
                    for item in res:
                        lbl = item.get("label", "").lower()
                        sc = float(item.get("score", 0.0))
                        if any(k in lbl for k in ("art", "fake", "ai", "syn")):
                            v_art = sc
                        elif any(k in lbl for k in ("hum", "real", "auth")):
                            v_hum = sc
                    vit_scores.append((v_art, v_hum))
            except Exception as e:
                logger.warning("Batch ViT error (%s), defaulting to 0.5", e)
                vit_scores = [(0.5, 0.5)] * len(batch_imgs)
        else:
            vit_scores = [(0.5, 0.5)] * len(batch_imgs)

        # 2. Multi-threaded physical forensics
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            forensic_results = list(pool.map(extract_forensics_only, batch_imgs))

        # 3. Assemble 14-D vectors
        for idx_in_batch, ((v_art, v_hum), forensics) in enumerate(zip(vit_scores, forensic_results)):
            feat_vec = np.concatenate([np.array([v_art, v_hum], dtype=np.float64), forensics])
            features_list.append(feat_vec)
            valid_indices.append(batch_img_indices[idx_in_batch])

        pct = min(100.0, (end_idx / total) * 100.0)
        print(f"      Processed {end_idx}/{total} images ({pct:.1f}%)")

    return features_list, valid_indices


def train_combined_model(
    dataset: str = "data_3",
    max_florida_real: int = 300,
    max_data2_real: int = 300,
    max_data2_fake: int = 300,
    max_web_ai: int = 0,
    test_eval_count: int = 50,
):
    """
    Main training routine supporting data_3, FLORIDA, and data_2 (CIFAKE) datasets.
    Dataset options:
      - 'data_3' (default): Trains directly on user-provided data_3 images (AI Art, Deepfakes, Real Art, Real Faces, Real Scenes).
      - 'combined': Combines data_3, FLORIDA, and data_2 for maximal domain generalization.
      - 'florida': Legacy training on FLORIDA + data_2.
    """
    print("=" * 70)
    print(f"  FactGuard AI — Multimodal Image Model Training Pipeline [{dataset.upper()}]")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Collect Dataset Samples
    print("\n[1/5] Sourcing balanced training samples across datasets...")

    all_real_train: List[Tuple[str, int]] = []
    all_fake_train: List[Tuple[str, int]] = []

    data2_fake_test, data2_real_test = load_data2_samples(DATA_2_DIR, split="test", max_per_class=test_eval_count)

    if dataset in ("data_3", "combined", "all"):
        data3_fake, data3_real = load_data3_samples(DATA_3_DIR)
        all_fake_train.extend([(p, 1) for p in data3_fake])
        all_real_train.extend([(p, 0) for p in data3_real])
        print(f"      Loaded data_3: {len(data3_fake)} AI/DeepFake, {len(data3_real)} Real images")

    if dataset in ("combined", "all", "florida"):
        florida_real = extract_florida_dataset(max_images=max_florida_real)
        data2_fake_train, data2_real_train = load_data2_samples(DATA_2_DIR, split="train", max_per_class=max_data2_real)
        all_real_train.extend([(p, 0) for p in florida_real])
        all_real_train.extend([(p, 0) for p in data2_real_train[:max_data2_real]])
        all_fake_train.extend([(p, 1) for p in data2_fake_train[:max_data2_fake]])
        if max_web_ai > 0:
            web_ai_paths = fetch_ai_generated_dataset(target_count=max_web_ai)
            all_fake_train.extend([(p, 1) for p in web_ai_paths])

    if not all_real_train or not all_fake_train:
        raise ValueError(f"Insufficient training samples collected (Real: {len(all_real_train)}, Fake: {len(all_fake_train)})")

    # Balance counts to prevent class imbalance skew
    target_len = min(len(all_real_train), len(all_fake_train))
    np.random.seed(42)
    real_indices = np.random.permutation(len(all_real_train))[:target_len]
    fake_indices = np.random.permutation(len(all_fake_train))[:target_len]
    all_real_train = [all_real_train[i] for i in real_indices]
    all_fake_train = [all_fake_train[i] for i in fake_indices]

    train_samples = all_real_train + all_fake_train
    sample_indices = np.random.permutation(len(train_samples))
    train_samples = [train_samples[i] for i in sample_indices]

    print(f"      Balanced Training Pool: {target_len} REAL vs {target_len} FAKE/AI/DEEPFAKE ({len(train_samples)} total)")

    # Step 2: Initialize ViT Backbone
    print("\n[2/5] Initializing deep Vision Transformer (ViT) backbone...")
    hf_pipe = _load_hf_image_classifier()

    # Step 3: Fast Batched Feature Extraction
    print("\n[3/5] Extracting 14-D multimodal features via batch tensor processing...")
    img_paths = [s[0] for s in train_samples]
    raw_labels = [s[1] for s in train_samples]

    features_list, valid_indices = extract_features_batch(img_paths, hf_pipe, batch_size=32)

    X = np.array(features_list, dtype=np.float64)
    y = np.array([raw_labels[i] for i in valid_indices], dtype=np.int32)
    print(f"      Feature matrix constructed: {X.shape} (Labels: {np.bincount(y)})")

    # Save feature cache for fast inspection & reproduction
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = DATA_3_CACHE_PATH if dataset == "data_3" else FEATURES_CACHE_PATH
    np.savez_compressed(cache_file, X=X, y=y)
    print(f"      Cached feature matrix to: {cache_file}")

    # Step 4: Strict Featurization Ordering & Model Training
    print("\n[4/5] Training Soft-Voting Ensemble (HistGradientBoosting + RF + ExtraTrees)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    ensemble = VotingClassifier(
        estimators=[
            ('hgb', HistGradientBoostingClassifier(
                max_iter=160,
                max_leaf_nodes=31,
                l2_regularization=1.2,
                random_state=42,
            )),
            ('rf', RandomForestClassifier(
                n_estimators=180,
                max_depth=10,
                min_samples_split=4,
                class_weight='balanced',
                random_state=42,
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=140,
                max_depth=10,
                min_samples_split=4,
                class_weight='balanced',
                random_state=42,
            )),
        ],
        voting='soft',
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(
            estimator=ensemble,
            cv=3,
            method='sigmoid',
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Validation evaluation
    y_pred_val = pipeline.predict(X_val)
    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_pred_val)
    val_roc = roc_auc_score(y_val, y_proba_val)

    print("\n" + "=" * 70)
    print(f"  INTERNAL HOLDOUT EVALUATION ({dataset.upper()} - 20% Stratified Split)")
    print("=" * 70)
    print(f"  Holdout Accuracy : {val_acc * 100:.2f}%")
    print(f"  ROC-AUC Score    : {val_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, target_names=["REAL", "AI-GENERATED/FAKE"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_val))

    # Step 5: External Evaluation on data_2/test split
    if data2_fake_test and data2_real_test:
        print("\n" + "=" * 70)
        print("  EXTERNAL BENCHMARK EVALUATION (data_2 CIFAKE Holdout Test Split)")
        print("=" * 70)
        test_samples = [(p, 0) for p in data2_real_test] + [(p, 1) for p in data2_fake_test]
        test_paths = [s[0] for s in test_samples]
        test_labels = [s[1] for s in test_samples]

        test_feats, test_valid_idx = extract_features_batch(test_paths, hf_pipe, batch_size=32)
        if test_feats:
            X_ext = np.array(test_feats, dtype=np.float64)
            y_ext = np.array([test_labels[i] for i in test_valid_idx], dtype=np.int32)
            y_ext_pred = pipeline.predict(X_ext)
            y_ext_proba = pipeline.predict_proba(X_ext)[:, 1]
            ext_acc = accuracy_score(y_ext, y_ext_pred)
            ext_roc = roc_auc_score(y_ext, y_ext_proba)
            print(f"  data_2/test Accuracy : {ext_acc * 100:.2f}%")
            print(f"  data_2/test ROC-AUC  : {ext_roc:.4f}")

    # Step 6: Save Model Artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, FLORIDA_MODEL_PATH)
    print(f"\n[+] Trained multimodal ensemble saved to: {FLORIDA_MODEL_PATH}")
    print(f"    Total elapsed training time: {time.time() - t_start:.2f}s")
    print("=" * 70 + "\n")
    return pipeline


train_florida_model = train_combined_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FactGuard AI — Image Verification Model Trainer")
    parser.add_argument("--dataset", choices=["data_3", "combined", "florida"], default="data_3",
                        help="Dataset to train on (default: data_3)")
    parser.add_argument("--eval-count", type=int, default=50,
                        help="Number of external test samples per class to evaluate on")
    args = parser.parse_args()

    train_combined_model(dataset=args.dataset, test_eval_count=args.eval_count)
