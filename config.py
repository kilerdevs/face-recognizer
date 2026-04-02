# ---------------------------------------------------------------------------
# config.py  –  all tunable constants
# ---------------------------------------------------------------------------

# InsightFace model.  buffalo_l = RetinaFace detector + ArcFace-R100 (best accuracy)
MODEL_NAME = "buffalo_l"

# Internal detection resolution fed to RetinaFace.
# Larger → catches smaller/partially-occluded faces, at the cost of VRAM/time.
DETECTION_SIZE = (992, 992)

# Images wider/taller than this are downscaled before detection.
# After detection, bbox is projected back to original coords.
MAX_DETECT_DIM = 4096

# Face thumbnail dimensions saved to disk (square, px)
FACE_THUMBNAIL_SIZE = 224

# Padding added around the detected bbox before cropping (relative to bbox size)
FACE_PADDING_RATIO = 0.38

# Annotated JPEG settings
OUTPUT_IMAGE_MAX_DIM = 2000
OUTPUT_JPEG_QUALITY  = 92

# Clustering algorithm: "agglomerative" | "hdbscan" | "dbscan"
CLUSTERING_ALGORITHM = "agglomerative"

# Agglomerative clustering:  cosine distance threshold.
# Lower = stricter (more identities, fewer false merges).
# Well-tested range: 0.30 – 0.50.
CLUSTER_DISTANCE_THRESHOLD = 0.50
CLUSTER_LINKAGE            = "average"   # 'average' is most accurate for faces

# HDBSCAN / DBSCAN: minimum faces required to form a cluster.
# Faces that fall below this density are left unassigned (noise).
CLUSTER_MIN_CLUSTER_SIZE = 2

# Minimum face size (shorter side of bbox in original-image pixels).
# Faces smaller than this are too low-resolution for reliable embedding.
MIN_FACE_SIZE_PX = 40

# Minimum RetinaFace/SCRFD detection confidence; lower-scoring faces are dropped
MIN_DET_SCORE = 0.65

# Minimum det_score for drawing boxes in annotated output / live view
ANNOT_MIN_DET_SCORE = 0.68

# SQLite filename placed inside the output directory
DB_FILENAME = "faces.db"

# How many log lines to keep in the GUI log window before rolling over
LOG_MAX_LINES = 3_000

# Raw / image extensions scanned recursively from the input folder
RAW_EXTENSIONS = {
    ".cr2", ".cr3",          # Canon
    ".nef", ".nrw",          # Nikon
    ".arw", ".srf", ".sr2",  # Sony
    ".raf",                   # Fujifilm
    ".dng",                   # Adobe universal RAW
    ".rw2",                   # Panasonic
    ".orf",                   # Olympus
    ".pef", ".ptx",          # Pentax
    ".srw",                   # Samsung
    ".x3f",                   # Sigma
    ".raw",
}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
