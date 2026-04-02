# Face Recognizer

> Automated face detection, embedding, and clustering across large photo libraries — with a simple desktop GUI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![InsightFace](https://img.shields.io/badge/Model-InsightFace%20buffalo__l-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

---

## Overview

Face Recognizer scans a folder of photos, detects every face using state-of-the-art deep learning models, generates a 512-dimensional embedding for each one, and automatically groups them into identities — no cloud, no account, no data leaving your machine.

It supports JPEG, TIFF, PNG, and a wide range of **camera RAW formats** (Canon, Nikon, Sony, Fujifilm, and more), making it suitable for photographers working with unprocessed files straight off the camera.

---

## Features

- **Face detection** via InsightFace `buffalo_l` (RetinaFace + ArcFace-R100), one of the most accurate open-source pipelines available
- **Embedding generation** — 512-d ArcFace vectors stored in a local SQLite database
- **Three clustering algorithms** to group faces into identities:
  - **Agglomerative** *(default)* — hierarchical, no noise labels, reliable on clean datasets
  - **HDBSCAN** — density-based, marks ambiguous/low-confidence faces as unassigned
  - **DBSCAN** — density-based with explicit epsilon neighbourhood control
- **RAW file support** — Canon (CR2/CR3), Nikon (NEF/NRW), Sony (ARW), Fujifilm (RAF), Adobe DNG, Panasonic (RW2), Olympus (ORF), Pentax (PEF), Samsung (SRW), Sigma (X3F), and more
- **GPU-accelerated inference** via ONNX Runtime with automatic CPU fallback
- **Face thumbnails** saved to disk alongside annotated versions of source images
- **PyQt5 desktop GUI** with live progress and log output
- **Self-bootstrapping** — missing Python packages are detected and installed automatically on first launch

---

## Requirements

| Requirement | Recommended |
|---|---|
| Python | 3.10 – 3.12 (3.13+ may lack pre-built ML wheels) |
| GPU | NVIDIA GPU with CUDA (CPU fallback works, but is slow) |
| RAM | 8 GB+ (16 GB+ recommended for large libraries) |
| OS | Windows / Linux / macOS |

---

## Installation

```bash
git clone https://github.com/kilerdevs/face-recognizer.git
cd face-recognizer

pip install -r requirements.txt
```

> **Note:** On first run, the app will automatically attempt to install any missing packages and will try to install `onnxruntime-gpu`. If no NVIDIA GPU is detected, it falls back to the CPU build automatically.

---

## Usage

```bash
python main.py
```

The GUI will open. From there:

1. **Select an input folder** containing your photos
2. **Select an output folder** where thumbnails, annotated images, and the database will be saved
3. Click **Process** to detect and embed all faces
4. Click **Cluster** to group faces into identities
5. Browse the results in the person gallery

---

## Configuration

All tunable parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `buffalo_l` | InsightFace model pack |
| `DETECTION_SIZE` | `(992, 992)` | Internal resolution fed to RetinaFace |
| `MAX_DETECT_DIM` | `4096` | Images larger than this are downscaled before detection |
| `FACE_THUMBNAIL_SIZE` | `224` | Saved thumbnail size (px, square) |
| `FACE_PADDING_RATIO` | `0.38` | Padding around detected bbox before cropping |
| `CLUSTERING_ALGORITHM` | `agglomerative` | `agglomerative` / `hdbscan` / `dbscan` |
| `CLUSTER_DISTANCE_THRESHOLD` | `0.50` | Cosine distance threshold for clustering |
| `CLUSTER_LINKAGE` | `average` | Linkage method for agglomerative clustering |
| `CLUSTER_MIN_CLUSTER_SIZE` | `2` | Min faces to form a cluster (HDBSCAN/DBSCAN) |
| `MIN_FACE_SIZE_PX` | `40` | Minimum face size in pixels (smaller faces are skipped) |
| `MIN_DET_SCORE` | `0.65` | Minimum RetinaFace detection confidence |

### Clustering tips

- **Agglomerative** with `threshold=0.40–0.50` is the safest starting point
- Lower the threshold for stricter separation (more identities, fewer false merges)
- Switch to **HDBSCAN** if your dataset is noisy — unrecognisable faces get left unassigned rather than forced into a cluster
- `CLUSTER_LINKAGE = "average"` generally outperforms `"complete"` or `"ward"` for face embeddings

---

## Tech Stack

| Layer | Library |
|---|---|
| GUI | PyQt5 |
| Face detection & embedding | InsightFace (`buffalo_l`) |
| Inference runtime | ONNX Runtime (GPU / CPU) |
| RAW decoding | rawpy + imageio |
| Image processing | OpenCV, Pillow |
| Clustering | scikit-learn (Agglomerative, HDBSCAN, DBSCAN) |
| Storage | SQLite (via Python `sqlite3`) |

---

## Project Structure

```
face-recognizer/
├── main.py          # Entry point — bootstraps dependencies, launches GUI
├── bootstrap.py     # Auto-installs missing packages, reports GPU/CUDA status
├── gui.py           # PyQt5 desktop interface
├── processor.py     # Face detection & embedding pipeline
├── clusterer.py     # Clustering algorithms & database update logic
├── database.py      # SQLite schema & queries
└── config.py        # All tunable constants
```

---

## GPU Support

The app prefers `onnxruntime-gpu` and checks for `CUDAExecutionProvider` at startup. If CUDA is not available, it automatically falls back to CPU inference — no manual configuration needed.

GPU status, VRAM, driver version, and CUDA version are printed to the console on every launch.

---

## Privacy

Everything runs **100% locally**. No images, embeddings, or metadata are ever sent to any external server.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) for details.
