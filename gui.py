# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# gui.py  –  PyQt5 GUI for the face recognition pipeline
# ---------------------------------------------------------------------------

import html as _html
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import (
    QObject, QSize, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QIcon, QImage, QPainter, QPalette, QPen, QPixmap,
    QTextCharFormat, QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from config import (
    ANNOT_MIN_DET_SCORE, CLUSTER_DISTANCE_THRESHOLD, DB_FILENAME,
    DETECTION_SIZE, IMAGE_EXTENSIONS, LOG_MAX_LINES, MIN_DET_SCORE,
)
from database import Database
from processor import ImageProcessor
from clusterer import FaceClusterer

logger = logging.getLogger(__name__)


# ===========================================================================
# Logging bridge  –  thread-safe forwarding of log records to the GUI
# ===========================================================================

class _LogBridge(QObject):
    """Singleton Qt object that carries log records across threads."""
    record_ready = pyqtSignal(str, int)   # (formatted_message, levelno)

_LOG_BRIDGE = _LogBridge()


class _GuiLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to _LOG_BRIDGE."""

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            _LOG_BRIDGE.record_ready.emit(msg, record.levelno)
        except Exception:
            pass


def _install_gui_log_handler():
    handler = _GuiLogHandler()
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                            datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logging.getLogger().addHandler(handler)


# ===========================================================================
# Helpers
# ===========================================================================

def _fmt_eta(secs: float) -> str:
    if secs < 0 or secs != secs:   # negative or NaN
        return "?"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s"
    h = secs // 3600
    m = (secs % 3600) // 60
    return f"{h}h {m:02d}m"


# ===========================================================================
# Widgets
# ===========================================================================

class EtaBar(QWidget):
    """Progress bar that shows percentage, ETA and throughput."""

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        if label:
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(90)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("color:#8ab4d4; font-size:10px;")
            lay.addWidget(lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(18)
        self.bar.setFormat("–")
        lay.addWidget(self.bar, stretch=3)

        self.eta_lbl   = QLabel("ETA: –")
        self.eta_lbl.setFixedWidth(110)
        self.eta_lbl.setStyleSheet("color:#8ab4d4; font-size:10px;")
        lay.addWidget(self.eta_lbl)

        self.speed_lbl = QLabel("")
        self.speed_lbl.setFixedWidth(100)
        self.speed_lbl.setStyleSheet("color:#999; font-size:10px;")
        lay.addWidget(self.speed_lbl)

        self._start: Optional[float] = None

    def start(self, indeterminate: bool = False):
        self._start = time.monotonic()
        if indeterminate:
            self.bar.setRange(0, 0)
            self.bar.setFormat("running…")
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
            self.bar.setFormat("0%")
        self.eta_lbl.setText("ETA: …")
        self.speed_lbl.setText("")

    def update(self, cur: int, total: int, unit: str = "item"):
        if total <= 0:
            return
        self.bar.setRange(0, total)
        self.bar.setValue(cur)
        pct = int(100 * cur / total)
        self.bar.setFormat(f"{pct}%  ({cur:,}/{total:,})")

        if self._start and cur > 0:
            elapsed  = time.monotonic() - self._start
            if elapsed == 0:
                return
            rate     = cur / elapsed
            remaining = (total - cur) / rate
            self.eta_lbl.setText(f"ETA: {_fmt_eta(remaining)}")
            self.speed_lbl.setText(f"{rate:.1f} {unit}/s")

    def finish(self, msg: str = "Done"):
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self.bar.setFormat(msg)
        if self._start:
            elapsed = time.monotonic() - self._start
            self.eta_lbl.setText(f"Took: {_fmt_eta(elapsed)}")
            self._start = None
        self.speed_lbl.setText("")

    def reset(self):
        self._start = None
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("–")
        self.eta_lbl.setText("ETA: –")
        self.speed_lbl.setText("")


class LogPanel(QTextEdit):
    """Coloured, scrollable, size-limited log panel."""

    _COLORS = {
        logging.DEBUG:    "#666",
        logging.INFO:     "#ccc",
        logging.WARNING:  "#ffa040",
        logging.ERROR:    "#ff6060",
        logging.CRITICAL: "#ff2020",
    }

    def __init__(self, max_lines: int = LOG_MAX_LINES, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas, Courier New, monospace", 9))
        self.setStyleSheet("background:#0d0d0d; border:1px solid #333;")
        self.max_lines  = max_lines
        self._line_count = 0

    def append_record(self, msg: str, level: int):
        color = self._COLORS.get(level, "#ccc")
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(msg + "\n", fmt)

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

        self._line_count += 1
        if self._line_count > self.max_lines:
            # Remove oldest 10 % to avoid constant trimming
            trim = self.max_lines // 10
            cur2 = self.textCursor()
            cur2.movePosition(QTextCursor.Start)
            cur2.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, trim)
            cur2.removeSelectedText()
            self._line_count -= trim

    def clear_log(self):
        self.clear()
        self._line_count = 0


class FaceThumb(QLabel):
    """Clickable face thumbnail with score badge and filename caption."""

    clicked = pyqtSignal(dict)

    THUMB_SIZE = 120

    def __init__(self, face: dict, parent=None):
        super().__init__(parent)
        sz = self.THUMB_SIZE
        self.setFixedSize(sz + 4, sz + 38)
        self.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QLabel{background:#252525;border:1px solid #3a3a3a;border-radius:4px;}"
            "QLabel:hover{border:1px solid #777;}"
        )
        self._face = face

        img_lbl = QLabel(self)
        img_lbl.setFixedSize(sz, sz)
        img_lbl.move(2, 2)
        tp = face.get("thumbnail_path", "")
        if tp and os.path.exists(tp):
            try:
                px = QPixmap(tp)
                if not px.isNull():
                    px = px.scaled(sz, sz, Qt.KeepAspectRatioByExpanding,
                                   Qt.SmoothTransformation)
                    pw, ph = px.width(), px.height()
                    img_lbl.setPixmap(
                        px.copy((pw - sz) // 2, (ph - sz) // 2, sz, sz)
                    )
                else:
                    img_lbl.setStyleSheet("background:#444;")
            except Exception:
                img_lbl.setStyleSheet("background:#444;")
        else:
            img_lbl.setStyleSheet("background:#333;")

        score  = face.get("det_score") or 0.0
        badge  = QLabel(f"{score:.2f}", self)
        badge.setFixedSize(36, 16)
        badge.move(sz - 34, 4)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            "background:rgba(0,0,0,170);color:#b0ffb0;"
            "font-size:8px;border-radius:3px;"
        )

        cap = QLabel(Path(face.get("filename", "")).stem, self)
        cap.setFixedSize(sz + 4, 32)
        cap.move(0, sz + 4)
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet("color:#888;font-size:8px;")
        cap.setWordWrap(True)

        self.setToolTip(
            f"Source: {face.get('filename', 'N/A')}\n"
            f"Detection score: {score:.4f}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self._face)


class FaceGrid(QScrollArea):
    face_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background:#1a1a1a;border:none;")
        self._container = QWidget()
        self._grid      = QGridLayout(self._container)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self.setWidget(self._container)

    def populate(self, faces: list):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        thumb_sz = FaceThumb.THUMB_SIZE + 14
        w        = self.viewport().width() or self.width()
        cols     = max(1, (w - 20) // thumb_sz)

        for i, face in enumerate(faces):
            th = FaceThumb(face)
            th.clicked.connect(self.face_clicked)
            self._grid.addWidget(th, i // cols, i % cols)


class ImageViewer(QLabel):
    """Scaled image viewer that preserves aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#111;border:1px solid #2a2a2a;")
        self._orig: Optional[QPixmap] = None
        self.setText("No image selected")
        self.setStyleSheet(
            "background:#111;border:1px solid #333;"
            "color:#555;font-size:13px;"
        )

    def set_image(self, path: str):
        if not path:
            self._orig = None
            self.setText("No image selected")
            return
        if not os.path.exists(path):
            self._orig = None
            self.setText(f"File not found:\n{path}")
            return
        px = QPixmap(path)
        if px.isNull():
            self._orig = None
            self.setText(f"Cannot load image:\n{path}")
            return
        self._orig = px
        self._fit()

    def _fit(self):
        if self._orig:
            scaled = self._orig.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()

    def set_image_with_faces(
        self,
        path: str,
        faces: list,
        orig_w: int,
        orig_h: int,
        min_score: float = 0.0,
    ):
        """Load base image and draw live bounding boxes from face data."""
        if not path:
            self._orig = None
            self.setText("No image selected")
            return
        if not os.path.exists(path):
            self._orig = None
            self.setText(f"File not found:\n{path}")
            return
        px = QPixmap(path)
        if px.isNull():
            self._orig = None
            self.setText(f"Cannot load image:\n{path}")
            return

        base_w, base_h = px.width(), px.height()
        scale = min(base_w / orig_w, base_h / orig_h) if orig_w > 0 and orig_h > 0 else 1.0

        canvas  = QPixmap(px)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        for face in faces:
            if face.get("det_score", 0) < min_score:
                continue
            x1 = max(0, int(face["x1"] * scale))
            y1 = max(0, int(face["y1"] * scale))
            x2 = min(base_w, int(face["x2"] * scale))
            y2 = min(base_h, int(face["y2"] * scale))
            if x2 <= x1 or y2 <= y1:
                continue

            pid = face.get("person_id")
            if pid:
                hue = int((pid * 137.508) % 360)
                col = QColor.fromHsv(hue, 210, 210)
            else:
                col = QColor(128, 128, 128)

            painter.setPen(QPen(col, 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            if face.get("name"):
                label = f"{face['name']} {face.get('surname') or ''}".strip()
            elif face.get("display_id"):
                label = str(face["display_id"])
            else:
                label = "?"
            label += f" {face.get('det_score', 0):.2f}"

            fsize = max(7, min(14, int((x2 - x1) / 8)))
            painter.setFont(QFont("Arial", fsize))
            fm   = painter.fontMetrics()
            tw   = fm.horizontalAdvance(label)
            th   = fm.height()
            bg_y = max(0, y1 - th - 2)
            painter.fillRect(x1, bg_y, tw + 6, th + 2, col)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x1 + 3, y1 - 3, label)

        painter.end()
        self._orig = canvas
        self._fit()


# ===========================================================================
# Worker threads
# ===========================================================================

class _ScanWorker(QThread):
    log      = pyqtSignal(str, int)
    count    = pyqtSignal(int)        # running total of files found
    finished = pyqtSignal(int, float) # (n_found, elapsed_s)

    def __init__(self, input_dir: str, db: Database):
        super().__init__()
        self.input_dir = input_dir
        self.db        = db

    def run(self):
        t0    = time.monotonic()
        found = 0
        errors = 0
        try:
            for root, _dirs, files in os.walk(self.input_dir):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                        fpath = os.path.join(root, fname)
                        try:
                            self.db.upsert_image(fname, fpath)
                            found += 1
                            if found % 25 == 0:
                                self.count.emit(found)
                                self.log.emit(
                                    f"Scanned {found} files…", logging.DEBUG
                                )
                        except Exception as exc:
                            self.log.emit(
                                f"WARN  upsert failed {fname}: {exc}", logging.WARNING
                            )
                            errors += 1
        except Exception as exc:
            self.log.emit(f"ERROR  scan failed: {exc}", logging.ERROR)

        elapsed = time.monotonic() - t0
        self.count.emit(found)
        msg = f"Scan complete: {found} image file(s) indexed  ({elapsed:.1f}s)"
        if errors:
            msg += f"  [{errors} error(s)]"
        self.log.emit(msg, logging.INFO)
        self.finished.emit(found, elapsed)


class _ProcessWorker(QThread):
    log      = pyqtSignal(str, int)
    progress = pyqtSignal(int, int)        # (cur, total)
    finished = pyqtSignal(int, int, float) # (images, faces, elapsed_s)

    def __init__(self, processor: ImageProcessor):
        super().__init__()
        self.processor = processor
        self._stop     = False

    def stop(self):
        self._stop = True

    def run(self):
        t0 = time.monotonic()
        try:
            imgs, faces = self.processor.process_all(
                log_cb      = self.log.emit,
                progress_cb = self.progress.emit,
                stop_flag   = lambda: self._stop,
            )
        except Exception as exc:
            self.log.emit(f"CRITICAL  process_all exception: {exc}", logging.CRITICAL)
            logger.exception("ProcessWorker unhandled exception")
            imgs, faces = 0, 0
        elapsed = time.monotonic() - t0
        self.finished.emit(imgs, faces, elapsed)


class _ClusterWorker(QThread):
    log      = pyqtSignal(str, int)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int, float) # (n_persons, elapsed_s)

    def __init__(self, clusterer: FaceClusterer):
        super().__init__()
        self.clusterer = clusterer

    def run(self):
        t0 = time.monotonic()
        try:
            n = self.clusterer.cluster(
                log_cb      = self.log.emit,
                progress_cb = self.progress.emit,
            )
        except Exception as exc:
            self.log.emit(f"CRITICAL  cluster exception: {exc}", logging.CRITICAL)
            logger.exception("ClusterWorker unhandled exception")
            n = 0
        elapsed = time.monotonic() - t0
        self.finished.emit(n, elapsed)


class _AnnotateWorker(QThread):
    log      = pyqtSignal(str, int)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int, float)

    def __init__(self, processor: ImageProcessor, min_score: float):
        super().__init__()
        self.processor = processor
        self.min_score = min_score

    def run(self):
        t0 = time.monotonic()
        try:
            n = self.processor.generate_annotated(
                log_cb      = self.log.emit,
                progress_cb = self.progress.emit,
                min_score   = self.min_score,
            )
        except Exception as exc:
            self.log.emit(f"CRITICAL  annotate exception: {exc}", logging.CRITICAL)
            logger.exception("AnnotateWorker unhandled exception")
            n = 0
        elapsed = time.monotonic() - t0
        self.finished.emit(n, elapsed)


# ===========================================================================
# Main window
# ===========================================================================

class MainWindow(QMainWindow):

    def __init__(self, env_info: Optional[dict] = None):
        super().__init__()
        self.env_info: dict          = env_info or {}
        self.db:        Optional[Database]       = None
        self.processor: Optional[ImageProcessor] = None
        self.clusterer: Optional[FaceClusterer]  = None
        self._active_worker: Optional[QThread]   = None
        self._person_id: Optional[int]           = None
        self._current_image: Optional[dict]      = None
        self._all_images:    List[dict]          = []
        self._all_persons:   List[dict]          = []
        self._last_out       = None
        self._last_det_size  = None
        self._last_model     = None

        self.setWindowTitle("Face Recognizer  ·  InsightFace GPU")
        self.resize(1500, 950)

        self._build_ui()
        self._apply_dark_theme()

        # Connect logging bridge to log panel
        _LOG_BRIDGE.record_ready.connect(self._on_log_record)
        _install_gui_log_handler()

        # Stats refresh timer
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(5_000)

        logger.info("GUI initialised  (env: %s)", self.env_info)
        self._refresh_sysinfo()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs)

        self.tabs.addTab(self._tab_processing(), "⚙  Processing")
        self.tabs.addTab(self._tab_people(),     "👤 People")
        self.tabs.addTab(self._tab_images(),     "🖼  Images")
        self.tabs.addTab(self._tab_stats(),      "📊 Statistics")

        self.tabs.currentChanged.connect(self._on_tab_change)
        self.setStatusBar(self.statusBar())
        self.statusBar().showMessage("Ready – select folders on the Processing tab")

    # ----------------------------------------------------------------
    # Processing tab
    # ----------------------------------------------------------------

    def _tab_processing(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        # --- system info ---
        sys_box = QGroupBox("System")
        sl = QGridLayout(sys_box)
        self.sys_gpu_lbl    = QLabel("GPU: detecting…")
        self.sys_cuda_lbl   = QLabel("CUDA: …")
        self.sys_ort_lbl    = QLabel("ORT: …")
        self.sys_model_lbl  = QLabel("Model: –")
        for i, lb in enumerate([self.sys_gpu_lbl, self.sys_cuda_lbl,
                                 self.sys_ort_lbl, self.sys_model_lbl]):
            lb.setStyleSheet("color:#8ab4d4; font-size:10px;")
            sl.addWidget(lb, 0, i)
        lay.addWidget(sys_box)

        # --- folders ---
        folder_box = QGroupBox("Folders")
        fl = QGridLayout(folder_box)
        fl.addWidget(QLabel("Input folder:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(
            "Root folder containing CR2/RAW images (searched recursively)…"
        )
        fl.addWidget(self.input_edit, 0, 1)
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(self._browse_input)
        fl.addWidget(b_in, 0, 2)

        fl.addWidget(QLabel("Output folder:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(
            "Where to write thumbnails, base JPEGs, annotated JPEGs and the database…"
        )
        fl.addWidget(self.output_edit, 1, 1)
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_output)
        fl.addWidget(b_out, 1, 2)
        lay.addWidget(folder_box)

        # --- settings ---
        set_box  = QGroupBox("Detection / Clustering Settings")
        set_grid = QGridLayout(set_box)
        set_grid.setHorizontalSpacing(8)
        set_grid.setVerticalSpacing(6)

        # ── Row 0: model, detection size, min detection score ────────────
        set_grid.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        _model_items = [
            ("buffalo_s  (fast, lower accuracy)",          "buffalo_s"),
            ("buffalo_l  (balanced, ArcFace-R100)  ★",    "buffalo_l"),
            ("antelopev2  (best accuracy, SCRFD-10G)",     "antelopev2"),
        ]
        for _lbl, _ in _model_items:
            self.model_combo.addItem(_lbl)
        self.model_combo.setCurrentIndex(1)   # buffalo_l default
        self._model_names = [m for _, m in _model_items]
        self.model_combo.setToolTip(
            "InsightFace model pack for detection and recognition:\n"
            "  buffalo_s    – MobileNet backbone.  Fastest, lowest VRAM, lower accuracy.\n"
            "  buffalo_l    – ArcFace-R100.  Best accuracy / speed balance  (default).\n"
            "  antelopev2   – SCRFD-10G detector + GlinTR100 recognition.\n"
            "                 Highest accuracy.  Downloads ~700 MB on first use.\n"
            "\nChanging the model forces the processor to reinitialise."
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        set_grid.addWidget(self.model_combo, 0, 1)

        set_grid.addWidget(QLabel("Detection size:"), 0, 2)
        self.det_combo = QComboBox()
        det_items = [
            ("320 × 320  (fast)", (320, 320)),
            ("480 × 480", (480, 480)),
            ("640 × 640", (640, 640)),
            ("800 × 800", (800, 800)),
            ("960 × 960", (960, 960)),
            ("992 × 992  (default)", (992, 992)),
            ("1120 × 1120  (high accuracy)", (1120, 1120)),
            ("1280 × 1280  (max)", (1280, 1280)),
        ]
        for label, _ in det_items:
            self.det_combo.addItem(label)
        self.det_combo.setCurrentIndex(5)
        self.det_combo.setToolTip(
            "Input resolution fed to the detector.\n"
            "Larger → catches smaller / partially occluded faces.\n"
            "Scales VRAM usage and processing time."
        )
        set_grid.addWidget(self.det_combo, 0, 3)
        self._det_sizes = [s for _, s in det_items]

        set_grid.addWidget(QLabel("Min det score:"), 0, 4)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.10, 0.99)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(MIN_DET_SCORE)
        self.score_spin.setDecimals(2)
        self.score_spin.setToolTip(
            "Detector confidence threshold.  Faces below this score are dropped.\n"
            "Lower to catch more marginal / partial face detections\n"
            "(may increase false positives)."
        )
        set_grid.addWidget(self.score_spin, 0, 5)
        set_grid.setColumnStretch(6, 1)

        # ── Row 1: cluster algo, threshold, linkage, annot score ─────────
        set_grid.addWidget(QLabel("Cluster algo:"), 1, 0)
        self.cluster_algo_combo = QComboBox()
        _algo_items = [
            ("Agglomerative  (hierarchical)  ★",  "agglomerative"),
            ("HDBSCAN  (density, noise-aware)",    "hdbscan"),
            ("DBSCAN  (density-based)",            "dbscan"),
        ]
        for _lbl, _ in _algo_items:
            self.cluster_algo_combo.addItem(_lbl)
        self._cluster_algos = [a for _, a in _algo_items]
        self.cluster_algo_combo.setToolTip(
            "Algorithm for grouping face embeddings into person identities:\n"
            "  Agglomerative – Hierarchical bottom-up.  Most reliable.  (recommended)\n"
            "  HDBSCAN       – Density-based.  Leaves ambiguous faces unassigned\n"
            "                  instead of forcing them into a cluster.  Good when\n"
            "                  collections have many solo / unclear shots.\n"
            "  DBSCAN        – Classic density clustering with fixed epsilon threshold."
        )
        self.cluster_algo_combo.currentIndexChanged.connect(
            self._on_cluster_algo_changed
        )
        set_grid.addWidget(self.cluster_algo_combo, 1, 1)

        set_grid.addWidget(QLabel("Threshold:"), 1, 2)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.20, 0.80)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(CLUSTER_DISTANCE_THRESHOLD)
        self.thresh_spin.setDecimals(3)
        self.thresh_spin.setToolTip(
            "Cosine distance threshold.  Meaning depends on algorithm:\n"
            "  Agglomerative – max intra-cluster distance.\n"
            "                  Lower = stricter (more persons, fewer false merges).\n"
            "  HDBSCAN       – cluster_selection_epsilon.  Helps merge nearby micro-clusters.\n"
            "  DBSCAN        – neighbourhood epsilon (max distance to be neighbours).\n"
            "Typical well-tuned range: 0.400 – 0.550."
        )
        set_grid.addWidget(self.thresh_spin, 1, 3)

        set_grid.addWidget(QLabel("Linkage:"), 1, 4)
        self.linkage_combo = QComboBox()
        for _lk in ["average", "complete", "single"]:
            self.linkage_combo.addItem(_lk)
        self.linkage_combo.setToolTip(
            "Linkage criterion for Agglomerative clustering (ignored by HDBSCAN / DBSCAN):\n"
            "  average  – mean distance between all cross-cluster pairs  (recommended).\n"
            "  complete – max distance between cross-cluster pairs  (tighter clusters).\n"
            "  single   – min distance  (tends to form chain clusters, use with caution)."
        )
        set_grid.addWidget(self.linkage_combo, 1, 5)

        set_grid.addWidget(QLabel("Annot. min score:"), 1, 6)
        self.annot_score_spin = QDoubleSpinBox()
        self.annot_score_spin.setRange(0.10, 0.99)
        self.annot_score_spin.setSingleStep(0.05)
        self.annot_score_spin.setValue(ANNOT_MIN_DET_SCORE)
        self.annot_score_spin.setDecimals(2)
        self.annot_score_spin.setToolTip(
            "Minimum det_score for drawing boxes in the annotated view\n"
            "and when generating annotated JPEGs.\n"
            "Raise to suppress low-confidence detections from the output."
        )
        self.annot_score_spin.valueChanged.connect(
            lambda _: self._reload_image_view()
        )
        set_grid.addWidget(self.annot_score_spin, 1, 7)
        set_grid.setColumnStretch(8, 1)

        lay.addWidget(set_box)

        # --- pipeline buttons ---
        pipe_box = QGroupBox("Pipeline")
        pipe_lay = QHBoxLayout(pipe_box)

        self.btn_scan    = QPushButton("1.  Scan Files")
        self.btn_process = QPushButton("2.  Detect Faces  (GPU)")
        self.btn_stop    = QPushButton("■  Stop")
        self.btn_cluster = QPushButton("3.  Cluster Faces")
        self.btn_annot   = QPushButton("4.  Generate Annotated JPEGs")
        self.btn_all     = QPushButton("▶  Run All Steps")
        self.btn_all.setStyleSheet("font-weight:bold; color:#8ab4d4;")
        self.btn_stop.setEnabled(False)

        self.btn_scan.setToolTip(
            "Walk the input folder recursively and register every image file in the database."
        )
        self.btn_process.setToolTip(
            "Run InsightFace on every unprocessed image; extract 512-dim ArcFace embeddings."
        )
        self.btn_stop.setToolTip("Request the current worker to stop after the current image.")
        self.btn_cluster.setToolTip(
            "Group all embeddings into person identities using Agglomerative Clustering.\n"
            "WARNING: resets existing person assignments."
        )
        self.btn_annot.setToolTip(
            "Redraw annotated JPEGs with current person names/IDs.\n"
            "Run this again after renaming people to refresh the output."
        )
        self.btn_all.setToolTip("Sequentially run all four pipeline steps.")

        self.btn_scan.clicked.connect(lambda: self._run_scan())
        self.btn_process.clicked.connect(lambda: self._run_process())
        self.btn_stop.clicked.connect(self._stop_worker)
        self.btn_cluster.clicked.connect(lambda: self._run_cluster())
        self.btn_annot.clicked.connect(lambda: self._run_annotate())
        self.btn_all.clicked.connect(self._run_all)

        for b in [self.btn_scan, self.btn_process, self.btn_stop,
                  self.btn_cluster, self.btn_annot, self.btn_all]:
            pipe_lay.addWidget(b)
        lay.addWidget(pipe_box)

        # --- per-step progress bars ---
        prog_box = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_box)
        self.eta_scan    = EtaBar("Scan")
        self.eta_process = EtaBar("Detect")
        self.eta_cluster = EtaBar("Cluster")
        self.eta_annot   = EtaBar("Annotate")
        for eb in [self.eta_scan, self.eta_process,
                   self.eta_cluster, self.eta_annot]:
            prog_lay.addWidget(eb)
        lay.addWidget(prog_box)

        # --- live stats strip ---
        stats_box = QGroupBox("Database Stats")
        stats_lay = QHBoxLayout(stats_box)
        self.stat_labels: Dict[str, QLabel] = {}
        for key in ["Images", "Processed", "Faces", "Persons", "Named", "Unassigned", "DB size"]:
            col = QVBoxLayout()
            title = QLabel(key)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color:#666;font-size:9px;")
            val = QLabel("–")
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet("color:#ddd;font-size:14px;font-weight:bold;")
            self.stat_labels[key] = val
            col.addWidget(title)
            col.addWidget(val)
            stats_lay.addLayout(col)
            if key != "DB size":
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet("color:#333;")
                stats_lay.addWidget(sep)
        lay.addWidget(stats_box)

        # --- log panel ---
        log_box = QGroupBox("Log")
        log_lay = QVBoxLayout(log_box)
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Live activity log:"))
        log_hdr.addStretch()
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(lambda: self.log_panel.clear_log())
        log_hdr.addWidget(btn_clear)
        lvl_combo = QComboBox()
        for lv in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            lvl_combo.addItem(lv)
        lvl_combo.setCurrentIndex(1)
        lvl_combo.currentIndexChanged.connect(
            lambda i: logging.getLogger().setLevel(
                [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR][i]
            )
        )
        log_hdr.addWidget(QLabel("Min level:"))
        log_hdr.addWidget(lvl_combo)
        log_lay.addLayout(log_hdr)

        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(200)
        log_lay.addWidget(self.log_panel)
        lay.addWidget(log_box, stretch=1)

        return w

    # ----------------------------------------------------------------
    # People tab
    # ----------------------------------------------------------------

    def _tab_people(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)

        splitter = QSplitter(Qt.Horizontal)
        lay.addWidget(splitter)

        # ---- left: person list ----
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(4, 4, 4, 4)
        left_l.setSpacing(4)

        # search + sort row
        search_row = QHBoxLayout()
        self.people_search = QLineEdit()
        self.people_search.setPlaceholderText("Search by name or ID…")
        self.people_search.textChanged.connect(self._filter_persons)
        search_row.addWidget(self.people_search)
        self.people_sort = QComboBox()
        self.people_sort.addItems(["By faces ↓", "By ID ↑", "By name ↑"])
        self.people_sort.currentIndexChanged.connect(self._sort_persons)
        search_row.addWidget(self.people_sort)
        left_l.addLayout(search_row)

        self.person_list = QListWidget()
        self.person_list.setIconSize(QSize(56, 56))
        self.person_list.setSpacing(2)
        self.person_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.person_list.currentItemChanged.connect(self._on_person_selected)
        left_l.addWidget(self.person_list)

        action_row = QHBoxLayout()
        self.btn_refresh_ppl = QPushButton("↺ Refresh")
        self.btn_refresh_ppl.clicked.connect(self._load_persons)
        self.btn_merge_ppl   = QPushButton("⇒ Merge")
        self.btn_merge_ppl.setToolTip(
            "Select 2 or more persons (Ctrl+Click / Shift+Click), then click Merge.\n"
            "Smart naming: named person wins over unnamed;\n"
            "if both named, the one with more faces keeps its name."
        )
        self.btn_merge_ppl.clicked.connect(self._merge_persons)
        self.btn_delete_ppl  = QPushButton("🗑 Delete")
        self.btn_delete_ppl.setToolTip(
            "Delete the selected person(s) and unassign all their faces.\n"
            "Faces remain in the database but are no longer assigned to anyone."
        )
        self.btn_delete_ppl.clicked.connect(self._delete_persons)
        action_row.addWidget(self.btn_refresh_ppl)
        action_row.addWidget(self.btn_merge_ppl)
        action_row.addWidget(self.btn_delete_ppl)
        left_l.addLayout(action_row)

        splitter.addWidget(left_w)

        # ---- right: detail panel ----
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(4, 4, 4, 4)

        # identity form
        id_box = QGroupBox("Identity")
        id_lay = QGridLayout(id_box)
        id_lay.addWidget(QLabel("Display ID:"), 0, 0)
        self.disp_id_lbl = QLabel("—")
        self.disp_id_lbl.setStyleSheet("font-weight:bold; color:#8ab4d4;")
        id_lay.addWidget(self.disp_id_lbl, 0, 1)
        id_lay.addWidget(QLabel("Faces:"), 0, 2)
        self.face_cnt_lbl = QLabel("–")
        id_lay.addWidget(self.face_cnt_lbl, 0, 3)
        id_lay.addWidget(QLabel("Avg confidence:"), 0, 4)
        self.avg_conf_lbl = QLabel("–")
        id_lay.addWidget(self.avg_conf_lbl, 0, 5)

        id_lay.addWidget(QLabel("First name:"), 1, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. John")
        id_lay.addWidget(self.name_edit, 1, 1)
        id_lay.addWidget(QLabel("Surname:"), 1, 2)
        self.surname_edit = QLineEdit()
        self.surname_edit.setPlaceholderText("e.g. Smith")
        id_lay.addWidget(self.surname_edit, 1, 3)

        save_btn = QPushButton("💾  Save Name")
        save_btn.clicked.connect(self._save_person_name)
        clear_btn = QPushButton("✕  Clear Name")
        clear_btn.clicked.connect(self._clear_person_name)
        id_lay.addWidget(save_btn,  1, 4)
        id_lay.addWidget(clear_btn, 1, 5)
        right_l.addWidget(id_box)

        # face grid
        self.face_grid = FaceGrid()
        self.face_grid.face_clicked.connect(self._on_face_thumb_clicked)
        right_l.addWidget(self.face_grid, stretch=1)

        splitter.addWidget(right_w)
        splitter.setSizes([300, 1200])
        return w

    # ----------------------------------------------------------------
    # Images tab
    # ----------------------------------------------------------------

    def _tab_images(self) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)

        splitter = QSplitter(Qt.Horizontal)
        lay.addWidget(splitter)

        # ---- left ----
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(4, 4, 4, 4)

        filter_row = QHBoxLayout()
        self.img_filter = QLineEdit()
        self.img_filter.setPlaceholderText("Filter by filename…")
        self.img_filter.textChanged.connect(self._filter_images)
        filter_row.addWidget(self.img_filter)
        left_l.addLayout(filter_row)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        left_l.addWidget(self.image_list)

        nav_row = QHBoxLayout()
        self.btn_img_prev = QPushButton("◀ Prev")
        self.btn_img_next = QPushButton("Next ▶")
        self.btn_img_prev.clicked.connect(self._prev_image)
        self.btn_img_next.clicked.connect(self._next_image)
        nav_row.addWidget(self.btn_img_prev)
        nav_row.addWidget(self.btn_img_next)
        left_l.addLayout(nav_row)

        splitter.addWidget(left_w)

        # ---- right ----
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(4, 4, 4, 4)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self.btn_view_annot = QPushButton("Annotated")
        self.btn_view_base  = QPushButton("Base (no boxes)")
        self.btn_view_annot.setCheckable(True)
        self.btn_view_base.setCheckable(True)
        self.btn_view_annot.setChecked(True)
        self._view_grp = QButtonGroup(right_w)
        self._view_grp.setExclusive(True)
        self._view_grp.addButton(self.btn_view_annot)
        self._view_grp.addButton(self.btn_view_base)
        self._view_grp.buttonClicked.connect(lambda _: self._reload_image_view())
        view_row.addWidget(self.btn_view_annot)
        view_row.addWidget(self.btn_view_base)
        view_row.addStretch()
        right_l.addLayout(view_row)

        self.img_viewer = ImageViewer()
        right_l.addWidget(self.img_viewer, stretch=1)

        # face list for current image
        faces_box = QGroupBox("Faces in this image")
        faces_lay = QVBoxLayout(faces_box)
        self.img_faces_list = QListWidget()
        self.img_faces_list.setMaximumHeight(120)
        self.img_faces_list.itemDoubleClicked.connect(self._on_img_face_dblclicked)
        faces_lay.addWidget(self.img_faces_list)
        right_l.addWidget(faces_box)

        self.img_info_lbl = QLabel()
        self.img_info_lbl.setStyleSheet("color:#888;font-size:10px;padding:3px;")
        self.img_info_lbl.setWordWrap(True)
        right_l.addWidget(self.img_info_lbl)

        splitter.addWidget(right_w)
        splitter.setSizes([320, 1180])
        return w

    # ----------------------------------------------------------------
    # Statistics tab
    # ----------------------------------------------------------------

    def _tab_stats(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        stat_btn_row = QHBoxLayout()
        refresh_btn = QPushButton("↺  Refresh Statistics")
        refresh_btn.clicked.connect(self._populate_stats_tab)
        stat_btn_row.addWidget(refresh_btn)
        chart_btn = QPushButton("📊  Generate Appearance Chart (PNG)")
        chart_btn.setToolTip(
            "Generate a bar-chart PNG showing how many distinct photos\n"
            "each person appears in.  Saved to the output folder."
        )
        chart_btn.clicked.connect(self._generate_appearance_chart)
        stat_btn_row.addWidget(chart_btn)
        stat_btn_row.addStretch()
        lay.addLayout(stat_btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent;border:none;")
        inner = QWidget()
        self._stats_inner_lay = QVBoxLayout(inner)
        self._stats_inner_lay.setSpacing(8)
        scroll.setWidget(inner)
        lay.addWidget(scroll, stretch=1)

        self._stats_labels: Dict[str, QLabel] = {}
        return w

    def _populate_stats_tab(self):
        """Build/refresh the contents of the Statistics tab."""
        if not self.db:
            return

        # clear previous widgets
        lay = self._stats_inner_lay
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stats = self.db.get_stats()
        top_persons = self.db.get_top_persons(15)
        top_images  = self.db.get_images_with_most_faces(10)

        def _section(title: str) -> QGroupBox:
            box = QGroupBox(title)
            lay.addWidget(box)
            return box

        def _kv(box: QGroupBox, rows: list):
            gl = QGridLayout(box)
            for r, (k, v) in enumerate(rows):
                kl = QLabel(str(k) + ":")
                kl.setStyleSheet("color:#888;")
                vl = QLabel(str(v))
                vl.setStyleSheet("color:#ddd;font-weight:bold;")
                gl.addWidget(kl, r, 0)
                gl.addWidget(vl, r, 1)

        # --- summary ---
        summ = _section("Summary")
        _kv(summ, [
            ("Total images",       f"{stats['total_images']:,}"),
            ("Processed images",   f"{stats['processed_images']:,}"),
            ("Pending",            f"{stats['total_images'] - stats['processed_images']:,}"),
            ("Total faces",        f"{stats['total_faces']:,}"),
            ("Total persons",      f"{stats['total_persons']:,}"),
            ("Named persons",      f"{stats['named_persons']:,}"),
            ("Unassigned faces",   f"{stats['unassigned_faces']:,}"),
            ("Database size",      f"{stats['db_size_mb']:.2f} MB"),
            ("Output directory",   self.output_edit.text() or "–"),
        ])

        # --- top persons ---
        if top_persons:
            tp_box = _section(f"Top {len(top_persons)} Persons by Face Count")
            tp_lay = QVBoxLayout(tp_box)
            max_cnt = top_persons[0]["face_count"] if top_persons else 1
            for p in top_persons:
                row_w = QWidget()
                row_l = QHBoxLayout(row_w)
                row_l.setContentsMargins(2, 2, 2, 2)
                name = ""
                if p["name"]:
                    name = f"  {p['name']} {p['surname'] or ''}".rstrip()
                id_lbl = QLabel(f"{p['display_id']}{name}")
                id_lbl.setFixedWidth(200)
                id_lbl.setStyleSheet("color:#8ab4d4;")
                row_l.addWidget(id_lbl)
                bar = QProgressBar()
                bar.setRange(0, max(max_cnt, 1))
                bar.setValue(p["face_count"])
                bar.setFormat(f"{p['face_count']} faces")
                bar.setFixedHeight(14)
                row_l.addWidget(bar, stretch=1)
                tp_lay.addWidget(row_w)

        # --- top images ---
        if top_images:
            ti_box = _section(f"Top {len(top_images)} Images by Face Count")
            ti_lay = QVBoxLayout(ti_box)
            max_cnt = top_images[0]["face_count"] if top_images else 1
            for img in top_images:
                row_w = QWidget()
                row_l = QHBoxLayout(row_w)
                row_l.setContentsMargins(2, 2, 2, 2)
                nm_lbl = QLabel(img["filename"])
                nm_lbl.setFixedWidth(280)
                nm_lbl.setStyleSheet("color:#aaa;")
                row_l.addWidget(nm_lbl)
                bar = QProgressBar()
                bar.setRange(0, max(max_cnt, 1))
                bar.setValue(img["face_count"])
                bar.setFormat(f"{img['face_count']} faces")
                bar.setFixedHeight(14)
                row_l.addWidget(bar, stretch=1)
                ti_lay.addWidget(row_w)

        # --- system / model ---
        env = self.env_info
        gpu = env.get("gpu", {})
        sys_box = _section("System / Model")
        _kv(sys_box, [
            ("Python",       env.get("python_ver", sys.version.split()[0])),
            ("Platform",     env.get("platform", "–")),
            ("ONNX Runtime", f"{env.get('ort_ver','?')}  [{env.get('ort_mode','?')}]"),
            ("GPU",          gpu.get("gpu_name", "not detected")),
            ("VRAM",         f"{gpu.get('vram_total_mb',0):,} MB  /  {gpu.get('vram_free_mb',0):,} MB free"
                             if gpu else "N/A"),
            ("CUDA version", gpu.get("cuda_version", "N/A") if gpu else "N/A"),
            ("Model",        self._model_name()),
        ])

        lay.addStretch()

    def _generate_appearance_chart(self):
        """Generate a bar-chart PNG of each person's appearance count in photos."""
        if not self.db:
            QMessageBox.warning(self, "No database", "Open a project first.")
            return
        out_dir = self.output_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Output folder", "Please select an output folder.")
            return

        data = self.db.get_person_image_counts()
        if not data:
            QMessageBox.information(self, "No data",
                                    "No persons found – run clustering first.")
            return

        # Cap to top 60 so the chart stays readable
        data = data[:60]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            # Prefer a font with full Latin-Extended (Polish) support
            _pref = ["DejaVu Sans", "Arial", "Calibri", "Segoe UI"]
            _avail = {f.name for f in fm.fontManager.ttflist}
            _family = next((f for f in _pref if f in _avail), None)
            if _family:
                matplotlib.rcParams["font.family"] = _family

            labels = []
            for p in data:
                if p.get("name"):
                    lbl = p["name"]
                    if p.get("surname"):
                        lbl += f" {p['surname']}"
                else:
                    lbl = p["display_id"]
                labels.append(lbl)

            counts = [p["image_count"] for p in data]

            fig_w = max(14, len(data) * 0.45)
            fig, ax = plt.subplots(figsize=(fig_w, 7))
            bars = ax.bar(range(len(labels)), counts, color="#2d5986", edgecolor="#1a3d60")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
            ax.set_ylabel("Number of photos", fontsize=11)
            ax.set_title("Person appearance count in photos", fontsize=13, pad=12)
            ax.bar_label(bars, padding=3, fontsize=8)
            ax.set_ylim(0, max(counts) * 1.12)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()

            out_path = os.path.join(out_dir, "appearance_chart.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Appearance chart saved: %s", out_path)
            QMessageBox.information(
                self, "Chart saved",
                f"Saved to:\n{out_path}"
            )
        except Exception as exc:
            logger.exception("generate_appearance_chart failed")
            QMessageBox.critical(self, "Chart error", str(exc))

    # ------------------------------------------------------------------
    # dark theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self):
        pal = QPalette()
        bg  = QColor(28, 28, 28)
        bg2 = QColor(42, 42, 42)
        txt = QColor(218, 218, 218)
        hi  = QColor(65, 125, 180)

        pal.setColor(QPalette.Window,          bg)
        pal.setColor(QPalette.WindowText,      txt)
        pal.setColor(QPalette.Base,            QColor(18, 18, 18))
        pal.setColor(QPalette.AlternateBase,   bg2)
        pal.setColor(QPalette.Text,            txt)
        pal.setColor(QPalette.Button,          bg2)
        pal.setColor(QPalette.ButtonText,      txt)
        pal.setColor(QPalette.Highlight,       hi)
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        pal.setColor(QPalette.ToolTipBase,     bg2)
        pal.setColor(QPalette.ToolTipText,     txt)
        QApplication.setPalette(pal)
        self.setStyleSheet("""
            QTabBar::tab          { padding: 7px 18px; font-size:11px; }
            QTabBar::tab:selected { background:#3c3c3c; color:#fff; }
            QGroupBox             { border:1px solid #3a3a3a; border-radius:4px;
                                    margin-top:8px; padding-top:6px; }
            QGroupBox::title      { subcontrol-origin:margin; left:8px;
                                    color:#6a9fbf; font-size:10px; }
            QPushButton           { padding:5px 14px; border-radius:3px;
                                    background:#3c3f41; border:1px solid #555; }
            QPushButton:hover     { background:#4c5258; }
            QPushButton:pressed   { background:#2d5986; }
            QPushButton:disabled  { color:#555; border-color:#333; }
            QPushButton:checked   { background:#2d5986; border-color:#5090c0; }
            QLineEdit, QTextEdit  { border:1px solid #4a4a4a; border-radius:3px;
                                    padding:3px; background:#1a1a1a; }
            QProgressBar          { border:1px solid #444; border-radius:3px;
                                    text-align:center; font-size:9px; }
            QProgressBar::chunk   { background:#2d5986; }
            QComboBox             { padding:3px 6px; background:#2a2a2a;
                                    border:1px solid #4a4a4a; border-radius:3px; }
            QListWidget           { border:1px solid #3a3a3a; }
            QScrollBar:vertical   { width:10px; background:#222; }
            QScrollBar::handle:vertical { background:#4a4a4a; border-radius:4px; min-height:20px; }
            QSplitter::handle     { background:#333; }
            QDoubleSpinBox        { padding:2px 4px; background:#1a1a1a;
                                    border:1px solid #4a4a4a; border-radius:3px; }
        """)

    # ------------------------------------------------------------------
    # System info refresh
    # ------------------------------------------------------------------

    def _refresh_sysinfo(self):
        env = self.env_info
        gpu = env.get("gpu", {})
        cuda_ok = env.get("ort_mode") == "gpu"

        self.sys_gpu_lbl.setText(
            f"GPU: {gpu.get('gpu_name','N/A')}"
            + (f"  ({gpu.get('vram_total_mb',0):,} MB)" if gpu else "")
        )
        cuda_style = "color:#60cc60;" if cuda_ok else "color:#cc6060;"
        self.sys_cuda_lbl.setText(
            ("✓  CUDA enabled" if cuda_ok else "✗  No CUDA (CPU mode)")
        )
        self.sys_cuda_lbl.setStyleSheet(cuda_style + "font-size:10px;")
        self.sys_ort_lbl.setText(f"ORT: {env.get('ort_ver','?')}  [{env.get('ort_mode','?')}]")
        self.sys_model_lbl.setText(f"Model: {self._model_name()}")

    # ------------------------------------------------------------------
    # browse buttons
    # ------------------------------------------------------------------

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder (CR2/RAW files)")
        if d:
            self.input_edit.setText(d)
            logger.info("Input folder: %s", d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_edit.setText(d)
            logger.info("Output folder: %s", d)

    # ------------------------------------------------------------------
    # init / re-init helpers
    # ------------------------------------------------------------------

    def _det_size(self) -> tuple:
        idx = max(0, min(self.det_combo.currentIndex(), len(self._det_sizes) - 1))
        return self._det_sizes[idx]

    def _model_name(self) -> str:
        idx = max(0, min(self.model_combo.currentIndex(), len(self._model_names) - 1))
        return self._model_names[idx]

    def _cluster_algo(self) -> str:
        idx = max(0, min(self.cluster_algo_combo.currentIndex(), len(self._cluster_algos) - 1))
        return self._cluster_algos[idx]

    def _on_model_changed(self, _idx: int):
        """Mark processor for reinit and update sysinfo label."""
        self._last_model = None
        self.sys_model_lbl.setText(f"Model: {self._model_name()}")

    def _on_cluster_algo_changed(self, _idx: int):
        """Enable linkage selector only for agglomerative."""
        self.linkage_combo.setEnabled(self._cluster_algo() == "agglomerative")

    def _ensure_init(self) -> bool:
        """Initialise DB / processor / clusterer. Returns True on success."""
        out_dir    = self.output_edit.text().strip()
        det_size   = self._det_size()
        model_name = self._model_name()

        if not out_dir:
            QMessageBox.warning(
                self, "Output folder", "Please select an output folder first."
            )
            return False

        need = (
            self.db is None
            or self._last_out != out_dir
            or self._last_det_size != det_size
            or self._last_model != model_name
        )
        if need:
            try:
                os.makedirs(out_dir, exist_ok=True)
                db_path        = os.path.join(out_dir, DB_FILENAME)
                self.db        = Database(db_path)
                self.processor = ImageProcessor(
                    out_dir, self.db,
                    det_size   = det_size,
                    model_name = model_name,
                )
                self.clusterer = FaceClusterer(
                    self.db,
                    distance_threshold = self.thresh_spin.value(),
                    algorithm          = self._cluster_algo(),
                    linkage            = self.linkage_combo.currentText(),
                )
                self._last_out      = out_dir
                self._last_det_size = det_size
                self._last_model    = model_name
                logger.info(
                    "Initialised  db=%s  det_size=%s  model=%s",
                    db_path, det_size, model_name,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Initialisation error", str(exc))
                logger.exception("Initialisation failed")
                return False

        # Always sync clusterer settings (no full reinit needed for these)
        self.clusterer.distance_threshold = self.thresh_spin.value()
        self.clusterer.algorithm          = self._cluster_algo()
        self.clusterer.linkage            = self.linkage_combo.currentText()
        return True

    # ------------------------------------------------------------------
    # busy / idle
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool):
        for b in (self.btn_scan, self.btn_process, self.btn_cluster,
                  self.btn_annot, self.btn_all):
            b.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        if not busy:
            self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _on_log_record(self, msg: str, level: int):
        self.log_panel.append_record(msg, level)

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def _update_stats(self):
        if not self.db:
            return
        try:
            s = self.db.get_stats()
        except Exception:
            return
        mapping = {
            "Images":     f"{s['total_images']:,}",
            "Processed":  f"{s['processed_images']:,}",
            "Faces":      f"{s['total_faces']:,}",
            "Persons":    f"{s['total_persons']:,}",
            "Named":      f"{s['named_persons']:,}",
            "Unassigned": f"{s['unassigned_faces']:,}",
            "DB size":    f"{s['db_size_mb']:.1f} MB",
        }
        for key, val in mapping.items():
            if key in self.stat_labels:
                self.stat_labels[key].setText(val)

    # ------------------------------------------------------------------
    # pipeline runners  (all use on_done callback for clean chaining)
    # ------------------------------------------------------------------

    def _run_scan(self, on_done=None):
        if not self._ensure_init():
            if on_done: on_done()
            return
        in_dir = self.input_edit.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Input folder",
                                "Please select a valid input folder.")
            if on_done: on_done()
            return

        self._set_busy(True)
        self.eta_scan.start(indeterminate=True)
        logger.info("=== SCAN  %s", in_dir)

        w = _ScanWorker(in_dir, self.db)
        w.log.connect(self._on_log_record)
        w.count.connect(lambda n: self.statusBar().showMessage(f"Scanning… {n} files found"))

        def _done(n, elapsed):
            self.eta_scan.finish(f"Found {n:,} files  ({elapsed:.1f}s)")
            self._update_stats()
            self.statusBar().showMessage(
                f"Scan complete: {n:,} files  ({elapsed:.1f}s)"
            )
            self._set_busy(False)
            if on_done: on_done()

        w.finished.connect(_done)
        self._active_worker = w
        w.start()

    def _run_process(self, on_done=None):
        if not self._ensure_init():
            if on_done: on_done()
            return

        self._set_busy(True)
        self.eta_process.start()
        total_known = self.db.get_unprocessed()
        total = len(total_known)
        logger.info("=== DETECT  %d image(s) queued", total)
        self.statusBar().showMessage(f"Detecting faces in {total:,} images…")

        w = _ProcessWorker(self.processor)
        w.log.connect(self._on_log_record)
        w.progress.connect(
            lambda cur, tot: self.eta_process.update(cur, tot, unit="img")
        )

        def _done(imgs, faces, elapsed):
            self.eta_process.finish(
                f"{imgs:,} images  ·  {faces:,} faces  ({elapsed:.0f}s)"
            )
            self._update_stats()
            self.statusBar().showMessage(
                f"Detection complete: {imgs:,} images, {faces:,} faces  "
                f"({elapsed:.0f}s)"
            )
            self._set_busy(False)
            if on_done: on_done()

        w.finished.connect(_done)
        self._active_worker = w
        w.start()

    def _run_cluster(self, on_done=None):
        if not self._ensure_init():
            if on_done: on_done()
            return
        stats = self.db.get_stats()
        if stats["total_persons"] > 0:
            r = QMessageBox.question(
                self, "Re-cluster",
                "Clustering will reset all current person assignments.\n"
                "Names you have assigned will be lost.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                if on_done: on_done()
                return

        self.clusterer.distance_threshold = self.thresh_spin.value()
        self._set_busy(True)
        self.eta_cluster.start()
        logger.info(
            "=== CLUSTER  algo=%s  threshold=%.3f  linkage=%s",
            self._cluster_algo(),
            self.thresh_spin.value(),
            self.linkage_combo.currentText(),
        )

        w = _ClusterWorker(self.clusterer)
        w.log.connect(self._on_log_record)
        w.progress.connect(
            lambda cur, tot: self.eta_cluster.update(cur, tot, unit="step")
        )

        def _done(n, elapsed):
            self.eta_cluster.finish(
                f"{n:,} person(s)  ({elapsed:.0f}s)"
            )
            self._update_stats()
            self.statusBar().showMessage(
                f"Clustering complete: {n:,} persons  ({elapsed:.0f}s)"
            )
            self._set_busy(False)
            if on_done: on_done()

        w.finished.connect(_done)
        self._active_worker = w
        w.start()

    def _run_annotate(self, on_done=None):
        if not self._ensure_init():
            if on_done: on_done()
            return

        self._set_busy(True)
        self.eta_annot.start()
        logger.info("=== ANNOTATE")

        w = _AnnotateWorker(self.processor, self.annot_score_spin.value())
        w.log.connect(self._on_log_record)
        w.progress.connect(
            lambda cur, tot: self.eta_annot.update(cur, tot, unit="img")
        )

        def _done(n, elapsed):
            self.eta_annot.finish(f"{n:,} annotated  ({elapsed:.0f}s)")
            self.statusBar().showMessage(
                f"Annotation complete: {n:,} JPEGs  ({elapsed:.0f}s)"
            )
            self._set_busy(False)
            if self.tabs.currentIndex() == 2:
                self._load_images()
            if on_done: on_done()

        w.finished.connect(_done)
        self._active_worker = w
        w.start()

    def _run_all(self):
        """Chain all four steps sequentially."""
        if not self._ensure_init():
            return
        in_dir = self.input_edit.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Input folder",
                                "Please select a valid input folder.")
            return
        logger.info("=== RUN ALL  (scan → detect → cluster → annotate)")
        for eta in [self.eta_scan, self.eta_process,
                    self.eta_cluster, self.eta_annot]:
            eta.reset()
        # Chain via nested callbacks
        self._run_scan(on_done=lambda: self._run_process(
            on_done=lambda: self._run_cluster(
                on_done=lambda: self._run_annotate()
            )
        ))

    def _stop_worker(self):
        if self._active_worker and self._active_worker.isRunning():
            if hasattr(self._active_worker, "stop"):
                self._active_worker.stop()
                logger.warning("Stop requested – waiting for current image to finish…")
                self.statusBar().showMessage(
                    "Stop requested – finishing current image…"
                )

    # ------------------------------------------------------------------
    # tab change
    # ------------------------------------------------------------------

    def _on_tab_change(self, idx: int):
        if idx == 1:
            self._load_persons()
        elif idx == 2:
            self._load_images()
        elif idx == 3:
            self._populate_stats_tab()

    # ------------------------------------------------------------------
    # People tab
    # ------------------------------------------------------------------

    def _load_persons(self):
        if not self.db:
            return
        try:
            self._all_persons = self.db.get_all_persons()
        except Exception as exc:
            logger.error("load_persons: %s", exc)
            return
        self._render_person_list(self._all_persons)

    def _render_person_list(self, persons: list):
        self.person_list.blockSignals(True)
        self.person_list.clear()
        for p in persons:
            full = ""
            if p["name"]:
                full = f"  {p['name']} {p['surname'] or ''}".rstrip()
            label = f"{p['display_id']}{full}\n{p['face_count']} face(s)"
            item  = QListWidgetItem(label)
            item.setData(Qt.UserRole, p["id"])
            rep = p.get("representative", "")
            if rep and os.path.exists(rep):
                try:
                    px = QPixmap(rep)
                    if not px.isNull():
                        item.setIcon(QIcon(
                            px.scaled(56, 56, Qt.KeepAspectRatioByExpanding,
                                      Qt.SmoothTransformation)
                        ))
                except Exception:
                    pass
            self.person_list.addItem(item)
        self.person_list.blockSignals(False)

    def _filter_persons(self, text: str):
        t = text.strip().lower()
        if not t:
            self._sort_persons()
            return
        filtered = [
            p for p in self._all_persons
            if t in (p.get("name") or "").lower()
            or t in (p.get("surname") or "").lower()
            or t in (p.get("display_id") or "").lower()
        ]
        self._render_person_list(filtered)

    def _sort_persons(self, _idx: int = -1):
        idx = self.people_sort.currentIndex()
        if idx == 0:
            self._render_person_list(
                sorted(self._all_persons, key=lambda p: -p["face_count"])
            )
        elif idx == 1:
            self._render_person_list(
                sorted(self._all_persons, key=lambda p: p["display_id"])
            )
        else:
            def _name_key(p):
                return (p.get("name") or "ZZZZZ", p.get("surname") or "")
            self._render_person_list(
                sorted(self._all_persons, key=_name_key)
            )

    def _on_person_selected(self, current, _prev):
        if not current or not self.db:
            return
        pid = current.data(Qt.UserRole)
        self._person_id = pid
        try:
            p = self.db.get_person(pid)
        except Exception as exc:
            logger.error("get_person(%s): %s", pid, exc)
            return
        if not p:
            return

        self.disp_id_lbl.setText(p.get("display_id", "?"))
        self.name_edit.setText(p.get("name") or "")
        self.surname_edit.setText(p.get("surname") or "")

        try:
            faces = self.db.get_faces_for_person(pid)
        except Exception as exc:
            logger.error("get_faces_for_person(%s): %s", pid, exc)
            faces = []

        # Face count and average confidence
        self.face_cnt_lbl.setText(str(len(faces)))
        if faces:
            avg_conf = sum(f.get("det_score", 0) for f in faces) / len(faces)
            self.avg_conf_lbl.setText(f"{avg_conf:.3f}")
        else:
            self.avg_conf_lbl.setText("–")

        self.face_grid.populate(faces)

    def _save_person_name(self):
        if not self.db or self._person_id is None:
            return
        name    = self.name_edit.text().strip()
        surname = self.surname_edit.text().strip()
        try:
            self.db.update_person_name(self._person_id, name, surname)
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))
            return
        display = self.disp_id_lbl.text()
        logger.info("Saved name: %s → %s %s", display, name, surname)
        self.statusBar().showMessage(f"Saved: {display} → {name} {surname}")
        # Update list item text in-place
        item = self.person_list.currentItem()
        if item:
            full  = f"  {name} {surname}".rstrip() if name else ""
            cnt   = self.face_cnt_lbl.text()
            item.setText(f"{display}{full}\n{cnt} face(s)")
        # Refresh the all_persons cache
        for p in self._all_persons:
            if p["id"] == self._person_id:
                p["name"]    = name
                p["surname"] = surname

    def _clear_person_name(self):
        self.name_edit.clear()
        self.surname_edit.clear()
        self._save_person_name()

    def _merge_persons(self):
        selected = self.person_list.selectedItems()
        if len(selected) < 2:
            QMessageBox.information(
                self, "Merge persons",
                "Select 2 or more persons (Ctrl+Click / Shift+Click), then click Merge."
            )
            return

        pids    = [item.data(Qt.UserRole) for item in selected]
        persons = [next((p for p in self._all_persons if p["id"] == pid), None)
                   for pid in pids]
        persons = [p for p in persons if p is not None]
        if len(persons) < 2:
            return

        # Survivor: named > unnamed; among equals, more faces wins.
        def _priority(p):
            return (bool(p.get("name")), p.get("face_count", 0))

        survivor = max(persons, key=_priority)
        others   = [p for p in persons if p["id"] != survivor["id"]]

        surv_label = survivor["display_id"]
        if survivor.get("name"):
            surv_label += f" ({survivor['name']} {survivor.get('surname','')})".rstrip()

        ids_str = ", ".join(p["display_id"] for p in others)
        r = QMessageBox.question(
            self, "Merge persons",
            f"Merge {len(others)+1} person(s) into:  {surv_label}\n\n"
            f"Absorbed: {ids_str}\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return

        try:
            for other in others:
                self.db.merge_persons(survivor["id"], other["id"])
        except Exception as exc:
            QMessageBox.critical(self, "Merge error", str(exc))
            return

        logger.info("Merged %d persons → %s", len(others), survivor["display_id"])
        self._load_persons()
        self._update_stats()

    def _delete_persons(self):
        selected = self.person_list.selectedItems()
        if not selected:
            return

        pids    = [item.data(Qt.UserRole) for item in selected]
        persons = [next((p for p in self._all_persons if p["id"] == pid), None)
                   for pid in pids]
        persons = [p for p in persons if p is not None]
        if not persons:
            return

        lines = []
        for p in persons:
            lbl = p["display_id"]
            if p.get("name"):
                lbl += f"  ({p['name']} {p.get('surname','')})".rstrip()
            lines.append(lbl)

        r = QMessageBox.question(
            self, "Delete persons",
            f"Delete {len(persons)} person(s) and unassign all their faces?\n\n"
            + "\n".join(lines[:15])
            + ("\n…" if len(lines) > 15 else "")
            + "\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return

        errors = 0
        for p in persons:
            try:
                self.db.delete_person(p["id"])
            except Exception as exc:
                logger.error("delete_person(%s): %s", p["id"], exc)
                errors += 1

        if errors:
            QMessageBox.warning(self, "Delete", f"{errors} deletion(s) failed – see log.")
        logger.info("Deleted %d person(s)", len(persons) - errors)
        self._person_id = None
        self._load_persons()
        self._update_stats()

    def _on_face_thumb_clicked(self, face: dict):
        """Switch to Images tab and show the source image with boxes."""
        self.tabs.setCurrentIndex(2)
        target = face.get("filename", "")
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            img  = item.data(Qt.UserRole) if item else None
            if img and img.get("filename") == target:
                self.image_list.setCurrentRow(row)
                return
        # fallback: just show the base image with the single known face box
        base_p = face.get("base_path", "")
        if base_p:
            orig_w = face.get("orig_width") or 0
            orig_h = face.get("orig_height") or 0
            self.img_viewer.set_image_with_faces(
                base_p, [face], orig_w, orig_h, self.annot_score_spin.value()
            )
            self.img_info_lbl.setText(
                f"Source: {target}  |  Score: {face.get('det_score', 0):.3f}"
            )

    # ------------------------------------------------------------------
    # Images tab
    # ------------------------------------------------------------------

    def _load_images(self):
        if not self.db:
            return
        try:
            self._all_images = self.db.get_all_images()
        except Exception as exc:
            logger.error("load_images: %s", exc)
            return
        self._filter_images(self.img_filter.text())

    def _filter_images(self, text: str):
        self.image_list.blockSignals(True)
        self.image_list.clear()
        t = text.strip().lower()
        for img in self._all_images:
            if t and t not in img["filename"].lower():
                continue
            status = f"  [{img['face_count']} face(s)]" if img["processed"] else "  [pending]"
            item   = QListWidgetItem(img["filename"] + status)
            item.setData(Qt.UserRole, img)
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)

    def _on_image_selected(self, current, _prev):
        if not current:
            return
        self._current_image = current.data(Qt.UserRole)
        self._reload_image_view()

    def _reload_image_view(self):
        img = self._current_image
        if not img:
            return
        base_p = img.get("base_path", "")

        use_annot = self.btn_view_annot.isChecked()
        if use_annot and base_p and self.db and img.get("id") and img.get("processed"):
            try:
                faces     = self.db.get_faces_for_image(img["id"])
                orig_w    = img.get("orig_width") or 0
                orig_h    = img.get("orig_height") or 0
                min_score = self.annot_score_spin.value()
                self.img_viewer.set_image_with_faces(base_p, faces, orig_w, orig_h, min_score)
            except Exception as exc:
                logger.error("_reload_image_view annotated: %s", exc)
                self.img_viewer.set_image(base_p)
        else:
            self.img_viewer.set_image(base_p)

        # Build info text
        parts = [
            img["filename"],
            f"{'Processed' if img['processed'] else 'Pending'}",
            f"{img['face_count']} face(s)",
            f"{img.get('orig_width','?')}×{img.get('orig_height','?')}",
        ]
        self.img_info_lbl.setText("  ·  ".join(parts))

        # Populate face list
        self.img_faces_list.clear()
        if self.db and img["processed"] and img.get("id"):
            try:
                faces = self.db.get_faces_for_image(img["id"])
                for face in faces:
                    name = ""
                    if face.get("name"):
                        name = f"  ({face['name']} {face.get('surname','')})".rstrip()
                    elif face.get("display_id"):
                        name = f"  ({face['display_id']})"
                    lbl  = (
                        f"Face  bbox=({face['x1']},{face['y1']})-"
                        f"({face['x2']},{face['y2']})"
                        f"  score={face.get('det_score',0):.2f}{name}"
                    )
                    self.img_faces_list.addItem(lbl)
            except Exception as exc:
                logger.error("get_faces_for_image: %s", exc)

    def _on_img_face_dblclicked(self, item: QListWidgetItem):
        """Navigate to the People tab for the person in this face."""
        # Could parse person ID from item; kept simple for now
        pass

    def _prev_image(self):
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def _next_image(self):
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, ev):
        if self.tabs.currentIndex() == 2:
            if ev.key() == Qt.Key_Left:
                self._prev_image()
                return
            if ev.key() == Qt.Key_Right:
                self._next_image()
                return
        super().keyPressEvent(ev)


# ===========================================================================
# entry point
# ===========================================================================

def run(env_info: Optional[dict] = None):
    # Guard against duplicate QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("Face Recognizer")
    app.setOrganizationName("bieniek")

    try:
        win = MainWindow(env_info=env_info)
        win.show()
        sys.exit(app.exec_())
    except Exception as exc:
        msg = f"Fatal error:\n{exc}"
        logger.exception("Fatal error in run()")
        try:
            QMessageBox.critical(None, "Fatal Error", msg)
        except Exception:
            print(msg, file=sys.stderr)
        sys.exit(1)
