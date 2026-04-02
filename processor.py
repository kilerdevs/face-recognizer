# ---------------------------------------------------------------------------
# processor.py  –  CR2/RAW loading + InsightFace GPU pipeline
# ---------------------------------------------------------------------------

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ANNOT_MIN_DET_SCORE,
    DETECTION_SIZE,
    FACE_PADDING_RATIO,
    FACE_THUMBNAIL_SIZE,
    MAX_DETECT_DIM,
    MIN_DET_SCORE,
    MIN_FACE_SIZE_PX,
    MODEL_NAME,
    OUTPUT_IMAGE_MAX_DIM,
    OUTPUT_JPEG_QUALITY,
    RAW_EXTENSIONS,
)
from database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# colour helper  (deterministic, visually spread per person)
# ---------------------------------------------------------------------------

def person_color(person_id: int) -> Tuple[int, int, int]:
    """Stable BGR colour based on person_id, maximally spread via golden angle."""
    hue = int((person_id * 137.508) % 180)   # OpenCV H ∈ [0,179]
    hsv = np.uint8([[[hue, 210, 210]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ---------------------------------------------------------------------------
# main processor class
# ---------------------------------------------------------------------------

class ImageProcessor:

    _PIL_FONT_CACHE: dict = {}   # size (int) → PIL ImageFont

    @classmethod
    def _get_pil_font(cls, size: int):
        """Return a PIL TrueType font that supports Unicode (incl. Polish)."""
        if size in cls._PIL_FONT_CACHE:
            return cls._PIL_FONT_CACHE[size]
        from PIL import ImageFont
        candidates = []
        try:
            import matplotlib as _mpl
            candidates.append(
                os.path.join(_mpl.get_data_path(),
                             "fonts", "ttf", "DejaVuSans.ttf")
            )
        except Exception:
            pass
        win_fonts = "C:/Windows/Fonts/"
        for _fn in ("arial.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf"):
            candidates.append(win_fonts + _fn)
        font = None
        for _path in candidates:
            if os.path.exists(_path):
                try:
                    font = ImageFont.truetype(_path, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        cls._PIL_FONT_CACHE[size] = font
        return font

    def __init__(
        self,
        output_dir: str,
        db: Database,
        det_size:   Tuple[int, int] = DETECTION_SIZE,
        model_name: str             = MODEL_NAME,
    ):
        self.output_dir = Path(output_dir)
        self.db         = db
        self.det_size   = det_size
        self.model_name = model_name
        self.faces_dir  = self.output_dir / "faces"
        self.base_dir   = self.output_dir / "base"
        self.annot_dir  = self.output_dir / "annotated"

        for d in (self.faces_dir, self.base_dir, self.annot_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.face_app = None
        self._init_model()

    # ------------------------------------------------------------------
    # model initialisation
    # ------------------------------------------------------------------

    def _init_model(self):
        """Load InsightFace with GPU if available, fall back to CPU."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed.  "
                "Run the app once to auto-install, or:  pip install insightface"
            ) from exc

        providers = []
        try:
            import onnxruntime as ort
            # On Windows: register every nvidia-* pip package's bin/ directory
            # into the Windows DLL search path before any CUDA provider is
            # touched.  ctypes.CDLL alone is not enough – the CUDA runtime
            # itself uses LoadLibraryW which requires the directory to be in
            # the search path (os.add_dll_directory).
            import sys as _sys
            if _sys.platform == "win32":
                import importlib as _il
                _nvidia_pkgs = [
                    "nvidia.cublas",
                    "nvidia.cufft",
                    "nvidia.curand",
                    "nvidia.cuda_nvrtc",
                    "nvidia.cuda_runtime",
                    "nvidia.cudnn",
                    "nvidia.cuda_cupti",
                    "nvidia.nvjitlink",
                ]
                for _pkg in _nvidia_pkgs:
                    try:
                        _mod = _il.import_module(_pkg)
                        _bin = os.path.join(os.path.dirname(_mod.__file__), "bin")
                        if os.path.isdir(_bin):
                            os.add_dll_directory(_bin)
                            logger.debug("DLL dir registered: %s", _bin)
                    except Exception:
                        pass
            try:
                ort.preload_dlls()
                logger.debug("onnxruntime.preload_dlls() succeeded")
            except Exception as _pdl_exc:
                logger.debug("onnxruntime.preload_dlls() skipped: %s", _pdl_exc)

            avail = {p.lower() for p in ort.get_available_providers()}
            if "cudaexecutionprovider" in avail:
                providers.append("CUDAExecutionProvider")
                logger.info("ONNX Runtime: CUDAExecutionProvider selected (RTX GPU)")
            else:
                logger.warning(
                    "ONNX Runtime: CUDAExecutionProvider NOT available – CPU mode.  "
                    "Install onnxruntime-gpu for GPU acceleration."
                )
        except ImportError:
            logger.warning("onnxruntime not importable – using CPU only")

        providers.append("CPUExecutionProvider")

        logger.info(
            "Loading InsightFace model '%s'  det_size=%s  providers=%s",
            self.model_name, self.det_size, providers,
        )
        self.face_app = FaceAnalysis(name=self.model_name, providers=providers)
        self.face_app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("InsightFace model '%s' ready", self.model_name)

    # ------------------------------------------------------------------
    # image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(filepath: str) -> Optional[np.ndarray]:
        """Return full-resolution RGB uint8 ndarray, or None on failure."""
        ext = Path(filepath).suffix.lower()
        if ext in RAW_EXTENSIONS:
            try:
                import rawpy
                with rawpy.imread(filepath) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=False,
                        output_bps=8,
                    )
                logger.debug(
                    "rawpy loaded %s  shape=%s", Path(filepath).name, rgb.shape
                )
                return rgb
            except Exception as exc:
                logger.error("rawpy failed on %s: %s", Path(filepath).name, exc)
                return None
        else:
            bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if bgr is None:
                logger.error("cv2.imread failed on %s", Path(filepath).name)
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downscale(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
        """Return (resized_img, scale_factor_to_original_coords)."""
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img, 1.0
        scale   = max_dim / max(h, w)
        new_w   = max(1, int(w * scale))
        new_h   = max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale

    @staticmethod
    def _crop_face(img_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop face from image with padding, resize to FACE_THUMBNAIL_SIZE."""
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        pad_x = bw * FACE_PADDING_RATIO
        pad_y = bh * FACE_PADDING_RATIO
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            logger.warning("_crop_face: degenerate bbox %s – returning blank", bbox)
            return np.zeros((FACE_THUMBNAIL_SIZE, FACE_THUMBNAIL_SIZE, 3), np.uint8)
        crop = img_rgb[cy1:cy2, cx1:cx2]
        return cv2.resize(
            crop,
            (FACE_THUMBNAIL_SIZE, FACE_THUMBNAIL_SIZE),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def _save_base(self, img_rgb: np.ndarray, stem: str) -> str:
        """Save downscaled base JPEG (no annotations). Returns absolute path."""
        h, w = img_rgb.shape[:2]
        if max(h, w) > OUTPUT_IMAGE_MAX_DIM:
            scale   = OUTPUT_IMAGE_MAX_DIM / max(h, w)
            out_img = cv2.resize(
                img_rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            out_img = img_rgb
        path = str(self.base_dir / f"{stem}.jpg")
        ok = cv2.imwrite(
            path,
            cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
        )
        if not ok:
            logger.warning("cv2.imwrite failed for base image: %s", path)
        return path

    # ------------------------------------------------------------------
    # single-image processing
    # ------------------------------------------------------------------

    def process_image(
        self,
        filepath: str,
        image_id: int,
        log_cb: Optional[Callable[[str, int], None]] = None,
    ) -> int:
        """
        Detect and embed all faces in one image file.
        log_cb receives (message, logging_level).
        Returns the number of accepted faces.
        """
        if self.face_app is None:
            raise RuntimeError("InsightFace model not initialised")

        name = Path(filepath).name
        stem = Path(filepath).stem
        t0   = time.perf_counter()

        def _log(msg, level=logging.INFO):
            logger.log(level, msg)
            if log_cb:
                log_cb(msg, level)

        _log(f"Loading  {name}", logging.DEBUG)

        # --- load ---
        rgb = self._load_image(filepath)
        if rgb is None:
            _log(f"SKIP  {name}  (image load failed)", logging.WARNING)
            self.db.mark_processed(image_id, 0, 0, 0, "")
            return 0

        orig_h, orig_w = rgb.shape[:2]
        _log(
            f"Loaded   {name}  {orig_w}×{orig_h}  "
            f"({orig_w * orig_h / 1_000_000:.1f} MP)",
            logging.DEBUG,
        )

        # --- save base (downscaled, unannotated) ---
        try:
            base_path = self._save_base(rgb, stem)
        except Exception as exc:
            _log(f"WARN  {name}: could not save base image – {exc}", logging.WARNING)
            base_path = ""

        # --- prepare for detection ---
        detect_rgb, scale_factor = self._downscale(rgb, MAX_DETECT_DIM)
        detect_bgr = cv2.cvtColor(detect_rgb, cv2.COLOR_RGB2BGR)

        _log(
            f"Detecting {name}  "
            f"(detect size {detect_bgr.shape[1]}×{detect_bgr.shape[0]},  "
            f"scale×{scale_factor:.2f})",
            logging.DEBUG,
        )

        # --- InsightFace ---
        try:
            raw_faces = self.face_app.get(detect_bgr)
        except Exception as exc:
            _log(f"ERROR  {name}: InsightFace failed – {exc}", logging.ERROR)
            self.db.mark_processed(image_id, 0, orig_w, orig_h, base_path)
            return 0

        # Filter by confidence score
        score_passed = [f for f in raw_faces if f.det_score >= MIN_DET_SCORE]
        # Filter by minimum face size in original-image coordinates
        faces = []
        for f in score_passed:
            bx1, by1, bx2, by2 = f.bbox * scale_factor
            bw = bx2 - bx1
            bh = by2 - by1
            if min(bw, bh) < MIN_FACE_SIZE_PX:
                _log(
                    f"  SKIP tiny face  ({bw:.0f}×{bh:.0f}px < {MIN_FACE_SIZE_PX}px)",
                    logging.DEBUG,
                )
                continue
            faces.append(f)

        _log(
            f"Detected {name}:  {len(raw_faces)} raw  →  "
            f"{len(score_passed)} score≥{MIN_DET_SCORE}  →  "
            f"{len(faces)} accepted  (size≥{MIN_FACE_SIZE_PX}px)",
            logging.DEBUG,
        )

        if not faces:
            self.db.mark_processed(image_id, 0, orig_w, orig_h, base_path)
            elapsed = time.perf_counter() - t0
            _log(f"  ✓  No accepted faces  {name}  ({elapsed:.1f}s)", logging.INFO)
            return 0

        # --- per-face processing ---
        face_dir = self.faces_dir / stem
        face_dir.mkdir(exist_ok=True)

        for i, face in enumerate(faces):
            # Scale bbox from detect-image → original-image coords
            try:
                bbox_orig = face.bbox * scale_factor
                # Clamp to image bounds
                bbox_orig[0] = max(0.0, bbox_orig[0])
                bbox_orig[1] = max(0.0, bbox_orig[1])
                bbox_orig[2] = min(float(orig_w), bbox_orig[2])
                bbox_orig[3] = min(float(orig_h), bbox_orig[3])
            except Exception as exc:
                _log(f"WARN  {name} face {i}: bbox error – {exc}", logging.WARNING)
                continue

            emb = face.embedding   # 512-dim float32, L2-normalised

            if emb is None or emb.shape[0] == 0:
                _log(f"WARN  {name} face {i}: empty embedding – skipping", logging.WARNING)
                continue

            # Crop from full-resolution original
            try:
                thumb_rgb  = self._crop_face(rgb, bbox_orig)
                thumb_path = str(face_dir / f"face_{i:04d}.jpg")
                ok = cv2.imwrite(
                    thumb_path,
                    cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 93],
                )
                if not ok:
                    _log(f"WARN  {name}: could not write thumbnail {thumb_path}", logging.WARNING)
                    thumb_path = ""
            except Exception as exc:
                _log(f"WARN  {name} face {i}: thumbnail error – {exc}", logging.WARNING)
                thumb_path = ""

            try:
                self.db.insert_face(
                    image_id=image_id,
                    bbox=bbox_orig,
                    embedding=emb,
                    det_score=float(face.det_score),
                    thumbnail_path=thumb_path,
                )
            except Exception as exc:
                _log(f"ERROR  {name} face {i}: DB insert failed – {exc}", logging.ERROR)

        self.db.mark_processed(image_id, len(faces), orig_w, orig_h, base_path)
        elapsed = time.perf_counter() - t0
        _log(
            f"  ✓  {len(faces)} face(s)  {name}  ({elapsed:.1f}s)",
            logging.INFO,
        )
        return len(faces)

    # ------------------------------------------------------------------
    # batch processing
    # ------------------------------------------------------------------

    def process_all(
        self,
        log_cb:      Optional[Callable[[str, int], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_flag:   Optional[Callable[[], bool]]         = None,
    ) -> Tuple[int, int]:
        """
        Process all unprocessed images in the database.
        Returns (images_processed, total_faces).
        """
        queue = self.db.get_unprocessed()
        total = len(queue)

        def _log(msg, level=logging.INFO):
            logger.log(level, msg)
            if log_cb:
                log_cb(msg, level)

        if total == 0:
            _log("No unprocessed images found in database.", logging.WARNING)
            if progress_cb:
                progress_cb(0, 0)
            return 0, 0

        _log(f"Processing {total} image(s)…", logging.INFO)

        images_done = 0
        faces_total = 0

        for idx, img_info in enumerate(queue):
            if stop_flag and stop_flag():
                _log("Processing stopped by user.", logging.WARNING)
                break

            if progress_cb:
                progress_cb(idx, total)

            try:
                n = self.process_image(img_info["filepath"], img_info["id"], log_cb)
                faces_total  += n
                images_done  += 1
            except Exception as exc:
                _log(
                    f"ERROR  {img_info.get('filename','?')}: unhandled exception – {exc}",
                    logging.ERROR,
                )
                logger.exception("Full traceback:")
                images_done += 1   # count as done so progress advances

        if progress_cb:
            progress_cb(images_done, total)

        _log(
            f"Processing complete:  {images_done}/{total} images,  "
            f"{faces_total} faces extracted.",
            logging.INFO,
        )
        return images_done, faces_total

    # ------------------------------------------------------------------
    # annotation pass
    # ------------------------------------------------------------------

    def generate_annotated(
        self,
        log_cb:      Optional[Callable[[str, int], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        min_score:   float = ANNOT_MIN_DET_SCORE,
    ) -> int:
        """
        Draw coloured bounding boxes + person labels on all base images.
        Returns number of annotated images written.
        """
        images = [
            img for img in self.db.get_all_images()
            if img["processed"] and img["base_path"]
        ]
        total = len(images)
        done  = 0

        def _log(msg, level=logging.INFO):
            logger.log(level, msg)
            if log_cb:
                log_cb(msg, level)

        _log(f"Annotating {total} image(s)…", logging.INFO)

        for idx, img_info in enumerate(images):
            if progress_cb:
                progress_cb(idx, total)

            base_path = img_info["base_path"]
            if not base_path or not os.path.exists(base_path):
                _log(
                    f"WARN  base image missing, skipping:  {img_info['filename']}",
                    logging.WARNING,
                )
                continue

            base_bgr = cv2.imread(base_path)
            if base_bgr is None:
                _log(
                    f"WARN  cv2.imread failed on base image:  {base_path}",
                    logging.WARNING,
                )
                continue

            base_h, base_w = base_bgr.shape[:2]
            orig_w = img_info.get("orig_width")  or base_w
            orig_h = img_info.get("orig_height") or base_h

            # Guard against divide-by-zero from DB nulls
            if orig_w <= 0 or orig_h <= 0:
                _log(
                    f"WARN  zero original dims for {img_info['filename']} – using base dims",
                    logging.WARNING,
                )
                orig_w, orig_h = base_w, base_h

            # Scale factor: original coords → base image coords
            scale = min(base_w / orig_w, base_h / orig_h)

            faces = self.db.get_faces_for_image(img_info["id"])

            # Convert to PIL for Unicode / Polish text support
            from PIL import Image as _PILImage, ImageDraw as _PILDraw
            pil_img  = _PILImage.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
            pil_draw = _PILDraw.Draw(pil_img)

            for face in faces:
                if face.get("det_score", 0) < min_score:
                    continue

                x1 = max(0, int(face["x1"] * scale))
                y1 = max(0, int(face["y1"] * scale))
                x2 = min(base_w, int(face["x2"] * scale))
                y2 = min(base_h, int(face["y2"] * scale))

                if x2 <= x1 or y2 <= y1:
                    continue   # degenerate box after scaling

                pid   = face["person_id"]
                color = person_color(pid) if pid else (128, 128, 128)

                # Label: "First Last"  or  "P0042"  or  "?"
                if face["name"]:
                    label = f"{face['name']} {face['surname'] or ''}".strip()
                elif face["display_id"]:
                    label = str(face["display_id"])
                else:
                    label = "?"

                full_label = label + f" {face['det_score']:.2f}"

                # PIL colour: BGR → RGB
                fill = (int(color[2]), int(color[1]), int(color[0]))

                # Font sized to face width
                fsize    = max(11, min(24, int((x2 - x1) / 9)))
                pil_font = self._get_pil_font(fsize)

                # Measure text
                try:
                    tb = pil_draw.textbbox((0, 0), full_label, font=pil_font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except AttributeError:
                    tw, th = pil_draw.textsize(full_label, font=pil_font)

                bg_y1 = max(0, y1 - th - 6)

                # Bounding box + label background + text
                pil_draw.rectangle([(x1, y1), (x2, y2)], outline=fill, width=2)
                pil_draw.rectangle([(x1, bg_y1), (x1 + tw + 6, y1)], fill=fill)
                pil_draw.text((x1 + 3, bg_y1 + 2), full_label,
                              fill=(255, 255, 255), font=pil_font)

            # Convert back to BGR for cv2.imwrite
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            stem     = Path(img_info["filename"]).stem
            out_path = self.annot_dir / f"{stem}_annotated.jpg"
            ok = cv2.imwrite(
                str(out_path), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
            )
            if ok:
                done += 1
            else:
                _log(f"WARN  cv2.imwrite failed: {out_path}", logging.WARNING)

            if idx % 50 == 0 or idx == total - 1:
                _log(
                    f"  Annotated {idx+1}/{total}  {img_info['filename']}",
                    logging.DEBUG,
                )

        if progress_cb:
            progress_cb(total, total)

        _log(
            f"Annotation complete:  {done}/{total} image(s) written.",
            logging.INFO,
        )
        return done
