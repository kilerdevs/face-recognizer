# ---------------------------------------------------------------------------
# clusterer.py  –  face clustering  (agglomerative / HDBSCAN / DBSCAN)
# ---------------------------------------------------------------------------

import logging
import time
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from config import (
    CLUSTER_DISTANCE_THRESHOLD,
    CLUSTER_LINKAGE,
    CLUSTER_MIN_CLUSTER_SIZE,
    CLUSTERING_ALGORITHM,
)
from database import Database

logger = logging.getLogger(__name__)


class FaceClusterer:

    def __init__(
        self,
        db: Database,
        distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
        algorithm:          str   = CLUSTERING_ALGORITHM,
        linkage:            str   = CLUSTER_LINKAGE,
        min_cluster_size:   int   = CLUSTER_MIN_CLUSTER_SIZE,
    ):
        self.db                 = db
        self.distance_threshold = distance_threshold
        self.algorithm          = algorithm
        self.linkage            = linkage
        self.min_cluster_size   = min_cluster_size

    # ------------------------------------------------------------------
    # internal clustering backends
    # ------------------------------------------------------------------

    def _cluster_agglomerative(
        self, dist_matrix: np.ndarray, n: int
    ) -> np.ndarray:
        """Agglomerative hierarchical clustering (no noise labels)."""
        if n == 1:
            return np.array([0], dtype=np.int32)
        model = AgglomerativeClustering(
            n_clusters         = None,
            distance_threshold = self.distance_threshold,
            metric             = "precomputed",
            linkage            = self.linkage,
        )
        return model.fit_predict(dist_matrix).astype(np.int32)

    def _cluster_hdbscan(
        self, dist_matrix: np.ndarray, n: int
    ) -> np.ndarray:
        """
        HDBSCAN density-based clustering.
        Faces that don't belong to any cluster are labelled -1 (noise).
        """
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise RuntimeError(
                "HDBSCAN requires scikit-learn ≥ 1.3.0.  "
                "Upgrade with: pip install -U scikit-learn"
            ) from exc

        if n == 1:
            return np.array([0], dtype=np.int32)

        model = HDBSCAN(
            min_cluster_size          = max(2, self.min_cluster_size),
            min_samples               = 1,
            metric                    = "precomputed",
            cluster_selection_epsilon = float(self.distance_threshold),
            cluster_selection_method  = "eom",
        )
        return model.fit_predict(dist_matrix).astype(np.int32)

    def _cluster_dbscan(
        self, dist_matrix: np.ndarray, n: int
    ) -> np.ndarray:
        """
        DBSCAN density-based clustering.
        Faces outside any neighbourhood are labelled -1 (noise).
        """
        from sklearn.cluster import DBSCAN

        if n == 1:
            return np.array([0], dtype=np.int32)

        model = DBSCAN(
            eps         = float(self.distance_threshold),
            min_samples = max(1, self.min_cluster_size),
            metric      = "precomputed",
        )
        return model.fit_predict(dist_matrix).astype(np.int32)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def cluster(
        self,
        log_cb:      Optional[Callable[[str, int], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Cluster all face embeddings and rebuild the persons table.

        Noise faces (HDBSCAN / DBSCAN label == -1) are left unassigned
        (person_id = NULL) rather than forced into a cluster.

        Returns the number of unique persons found.
        """
        STEPS = 5

        def _log(msg, level=logging.INFO):
            logger.log(level, msg)
            if log_cb:
                log_cb(msg, level)

        def _prog(step):
            if progress_cb:
                progress_cb(step, STEPS)

        _prog(0)
        _log("Loading face embeddings…", logging.INFO)

        face_data = self.db.get_all_embeddings()
        if not face_data:
            _log("No face embeddings found – run processing first.", logging.WARNING)
            _prog(STEPS)
            return 0

        face_ids   = [f[0] for f in face_data]
        embeddings = np.array([f[1] for f in face_data], dtype=np.float32)
        n          = len(embeddings)

        _log(f"Loaded {n} embeddings  (dim={embeddings.shape[1]})", logging.INFO)
        _prog(1)

        # --- pairwise cosine distance matrix ---
        mb = n * n * 8 / 1_048_576
        _log(
            f"Computing {n}×{n} cosine distance matrix  ({mb:.0f} MB)…",
            logging.INFO,
        )
        t0          = time.perf_counter()
        dist_matrix = cosine_distances(embeddings).astype(np.float64)
        np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)
        elapsed_dist = time.perf_counter() - t0
        nonzero = dist_matrix[dist_matrix > 0]
        _log(
            f"Distance matrix done in {elapsed_dist:.1f}s  "
            f"(min={nonzero.min():.3f}  median={np.median(dist_matrix):.3f})",
            logging.DEBUG,
        )
        _prog(2)

        # --- clustering ---
        algo = self.algorithm
        extra = (
            f"linkage={self.linkage}"
            if algo == "agglomerative"
            else f"min_cluster_size={self.min_cluster_size}"
        )
        _log(
            f"Running {algo} clustering  "
            f"(threshold={self.distance_threshold:.3f}  {extra})…",
            logging.INFO,
        )
        t1 = time.perf_counter()

        if algo == "hdbscan":
            labels = self._cluster_hdbscan(dist_matrix, n)
        elif algo == "dbscan":
            labels = self._cluster_dbscan(dist_matrix, n)
        else:
            labels = self._cluster_agglomerative(dist_matrix, n)

        elapsed_clust = time.perf_counter() - t1

        # Noise faces have label == -1; don't count them as persons
        n_noise   = int((labels == -1).sum())
        valid_lbl = labels[labels >= 0]
        n_persons = int(valid_lbl.max()) + 1 if len(valid_lbl) > 0 else 0

        if n_persons > 0:
            sizes = np.bincount(valid_lbl)
            _log(
                f"Clustering done in {elapsed_clust:.1f}s  →  "
                f"{n_persons} person(s)  "
                f"(largest: {sizes.max()} faces  "
                f"singletons: {(sizes == 1).sum()}  "
                f"noise/unassigned: {n_noise})",
                logging.INFO,
            )
        else:
            _log(
                f"Clustering done in {elapsed_clust:.1f}s  →  "
                f"0 persons (all {n_noise} face(s) are noise – "
                f"try lowering the threshold or switching algorithm)",
                logging.WARNING,
            )
        _prog(3)

        # --- update database ---
        _log("Resetting and rebuilding persons table…", logging.INFO)
        self.db.reset_clustering()

        label_to_pid: dict = {}
        for label in range(n_persons):
            display_id          = f"P{label + 1:04d}"
            pid                 = self.db.create_person(display_id)
            label_to_pid[label] = pid

        # Noise faces (label == -1) are intentionally excluded → person_id stays NULL
        pairs = [
            (face_ids[i], label_to_pid[int(labels[i])])
            for i in range(n)
            if labels[i] >= 0
        ]
        self.db.assign_persons(pairs)

        # Update statistics (face count + representative thumbnail)
        for label in range(n_persons):
            pid   = label_to_pid[label]
            faces = self.db.get_faces_for_person(pid)
            rep   = faces[0]["thumbnail_path"] if faces else ""
            self.db.update_person_meta(pid, len(faces), rep)

        _prog(STEPS)
        noise_note = f"  ({n_noise} face(s) left unassigned)" if n_noise else ""
        _log(
            f"Clustering complete:  {n_persons} person(s)  "
            f"from {n} face(s){noise_note}.",
            logging.INFO,
        )
        return n_persons
