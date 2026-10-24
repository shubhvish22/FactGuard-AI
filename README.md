
# FactGuard AI — Multimodal Fake News & Image Verification System

FactGuard AI is an enterprise-grade multimodal fact-checking and media forensics platform that verifies submitted **text claims** and **digital images**. It features a modern **Chrome Extension (Manifest V3)** coupled with a high-performance **Python Flask / Waitress WSGI backend**.

![Version](https://img.shields.io/badge/version-1.3.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      Chrome Extension (Manifest V3)                    │
│   • Popup UI: Text claims, image analysis, claim breakdown, history    │
│   • Background Service Worker: Context menus, tab messaging, badges   │
│   • Content Script: Inline image scanning, badge overlay, visual cues  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST / JSON (CORS Restricted)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Flask / Waitress WSGI Backend (Port 5000)            │
│   • app.py: Production REST endpoints, strict input validation         │
│   • security.py: SSRF protection, private IP filters, bounded requests │
│   • history_manager.py: SQLite WAL persistence & audit trails          │
│   • explanation_engine.py: XAI natural language reasoning              │
├────────────────────────────────────────────────────────────────────────┤
│                      Verification Engines                              │
│   • Text Pipeline: BERT Transformer + TF-IDF fallback + RSS Grounding │
│   • Image Pipeline: 64-bit pHash + ViT + FLORIDA Calibrated Ensemble   │
│   • Claim Extractor: Proposition decomposition & search query synthesis│
│   • Evidence Retriever: 5-Tier authoritative web evidence discovery    │
│   • Cross-Referencer: 5-Status verdicts (Supported, Contradicted, etc.)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Cache-First / Zero-Cost Fallback
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Models, Datasets & Storage                       │
│   • Text Transformer: jy46604790/Fake-News-Bert-Detect (cached)       │
│   • Vision Transformer: umm-maybe/AI-image-detector (cached)           │
│   • FLORIDA Ensemble: Calibrated RandomForestClassifier                │
│   • Baseline Fallback: TF-IDF + Logistic Regression                    │
│   • Database: SQLite (image_hashes.db) with WAL mode concurrency       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Factual Claim Extraction & Verification
- **Granular Decomposition**: Separates compound paragraphs into discrete, independently verifiable propositions.
- **Context Preservation**: Preserves original wording, qualifiers, and surrounding context.
- **Targeted Query Generation**: Synthesizes keyword-optimized search queries for each claim.
- **5-Tier Authoritative Evidence Hierarchy**:
  - **Tier 1**: Official government publications and original documents (`.gov`, `.mil`, WHO, UN).
  - **Tier 2**: Primary research, institutional reports, and court records (`.edu`, Nature, PubMed).
  - **Tier 3**: Established fact-checking organizations (Snopes, PolitiFact, Full Fact, BoomLive).
  - **Tier 4**: Reputable international news publications (Reuters, AP News, BBC, NYT).
  - **Tier 5**: General web publications (clearly tagged as lower-priority).
- **5 Standard Verification Statuses**:
  - `Supported`
  - `Contradicted`
  - `Mixed Evidence`
  - `Insufficient Evidence`
  - `Not Verifiable`

### 2. Multimodal Image Forensics
- **Perceptual Hashing (pHash)**: 64-bit perceptual hash with Hamming distance similarity to detect viral reuse.
- **Hugging Face ViT AI Image Detector**: Deep learning detection using `umm-maybe/AI-image-detector`.
- **FLORIDA Calibrated Ensemble**: Multimodal RandomForest trained on the FLORIDA dataset to distinguish authentic photos from synthetic generation.
- **Forensic Heuristic Sensors**:
  - Multi-quality Error Level Analysis (ELA at JPEG quality 75, 90, 95).
  - Laplacian variance noise uniformity.
  - Patch-level texture variation.
  - Color distribution entropy.
  - FFT 2D frequency domain energy distribution.
  - JPEG 8x8 blocking seam artifact analysis.
- **Reverse Image Search**: Google Cloud Vision Web Detection and SerpApi Google Lens integration with offline simulation fallback.
- **Optical Character Recognition (OCR)**: Google Vision text extraction from memes, screenshots, and infographics.

### 3. Persistent History & Audit Trails
- Local SQLite database (`image_hashes.db`) configured with **WAL (Write-Ahead Logging)** mode for multi-threaded concurrency.
- Full verification logs containing original text/image, claim propositions, retrieved source URLs, and timestamps.
- Built-in search, filtering (`all`, `text`, `image`), and pagination (`/history`).

### 4. Security & Hardening
- **SSRF Defense**: Strict validation rejecting loopback (`127.0.0.0/8`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local/cloud metadata (`169.254.169.254`), and carrier-grade NAT.
- **CORS Protection**: Access strictly restricted to Chrome extension IDs and local loopback origins.
- **Payload Bounds**: 16 MB max image upload; 50,000 character maximum text limit; bounded request timeouts.
- **Zero Secret Exposure**: API credentials stored exclusively in `backend/.env`.

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.11+
- Google Chrome (or Chromium-based browser)

### 2. Backend Setup
```bash
# Navigate to the backend directory
cd "fake news detector/backend"

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Configure environment variables
copy .env.example .env
# Edit .env to add your optional API keys (or leave empty for offline mode)

# Start the backend server (runs on http://127.0.0.1:5000)
python app.py
```

### 3. Chrome Extension Setup
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle in the upper right.
3. Click **Load unpacked**.
4. Select the `extension/` folder inside the project directory.
5. The FactGuard AI shield icon will appear in your Chrome toolbar.

---

## Configuration & Environment Variables

Create a `backend/.env` file based on `backend/.env.example`:

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `FLASK_HOST` | Host address to bind the backend server | `127.0.0.1` |
| `FLASK_PORT` | Port number for the REST API | `5000` |
| `FLASK_DEBUG` | Enable debug mode (`true` / `false`) | `false` |
| `HUGGINGFACE_API_TOKEN` | Hugging Face Access Token for authenticated Hub caching | Optional |
| `HF_TOKEN` | Alias for Hugging Face Token | Optional |
| `VISION_API_KEY` | Google Cloud Vision API Key (Web Detection & OCR) | Optional |
| `SERPAPI_API_KEY` | SerpApi Key for Google Lens Visual Matches | Optional |
| `FACTCHECK_API_KEY` | Google Fact Check Tools API Key | Optional |
| `GEMINI_API_KEY` | Google Gemini API Key for grounded web search | Optional |

> **Note**: FactGuard AI operates fully offline if no external API keys are configured. It utilizes local PyTorch transformers, local SQLite databases, and Google News RSS feeds at zero cost.

---

## REST API Reference

### Health Check
`GET /health`
Returns service status and engine availability.
```json
{
  "service": "FactGuard AI",
  "status": "ok",
  "version": "1.3.0",
  "engines": {
    "text_verifier": true,
    "text_model": "bert",
    "image_verifier": true,
    "ai_image_detector": true,
    "evidence_retrieval": true,
    "claim_extractor": true,
    "history_db": true
  }
}
```

### Text Verification & Fact-Checking
`POST /verify`
Payload:
```json
{
  "text": "The European Union announced updated climate targets for 2030.",
  "source_url": "https://example.com/article",
  "page_title": "EU Climate Update"
}
```

### Image Verification
`POST /upload-image` (Multipart `multipart/form-data`)
Form fields:
- `image`: Image binary file
- `source_url`: Optional source URL
- `webpage_title`: Optional title

`POST /verify-image-url`
Payload:
```json
{
  "url": "https://upload.wikimedia.org/wikipedia/commons/example.jpg"
}
```

### Verification History
`GET /history?page=1&limit=10&type=all&q=keyword`
`GET /history/<id>`
`DELETE /history/<id>`

---

## Running the Automated Test Suite

FactGuard AI includes an automated unit and pipeline regression suite:

```bash
# From the project root or backend directory:
python backend/test_suite.py
```

### Test Coverage Highlights:
- **BERT Guard & Decision Fusion**: Preserving high-confidence ML against weak external signals.
- **Claim Proposition Decomposition**: Regex & NLP extraction of multiple claims.
- **Multi-Tier Evidence Retrieval**: Prioritizing Tier 1/2 government & academic sources.
- **Image Forensics**: pHash collision avoidance, ELA, Laplacian noise, and FLORIDA ensemble scoring.
- **SSRF Protection**: Blocking RFC1918, loopback, and cloud metadata IPs.
- **Database CRUD**: Concurrency, pagination, search, and history deletion.

---

## License

MIT License — Copyright (c) FactGuard AI Contributors.
=======
# FactGuard-AI
AI-powered fake news detection and credibility verification for text and images.

