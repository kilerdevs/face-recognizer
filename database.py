# ---------------------------------------------------------------------------
# database.py  –  SQLite persistence with portable relative-path storage
# ---------------------------------------------------------------------------

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Database:
    """
    All file paths stored in the database are kept RELATIVE to ``output_dir``
    (which is the parent directory of the DB file itself).  This makes the
    entire output folder portable across machines and operating systems as
    long as the folder structure is preserved.
    """

    def __init__(self, db_path: str):
        self.db_path    = os.path.abspath(db_path)
        self.output_dir = str(Path(self.db_path).parent)
        self._init_schema()
        logger.debug("Database opened: %s  (output_dir=%s)", self.db_path, self.output_dir)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------

    def _rel(self, path: str) -> str:
        """Convert absolute path to relative (relative to output_dir)."""
        if not path:
            return ""
        path = os.path.abspath(path)
        try:
            rel = os.path.relpath(path, self.output_dir)
            # On Windows, relpath can produce paths starting with ".." when
            # the path is on a different drive.  Fall back to absolute in that case.
            if rel.startswith("..") and os.path.splitdrive(path)[0] != \
               os.path.splitdrive(self.output_dir)[0]:
                return path
            return rel
        except ValueError:
            # Different drive on Windows
            return path

    def _abs(self, rel_or_abs: str) -> str:
        """Resolve a stored path to an absolute path."""
        if not rel_or_abs:
            return ""
        if os.path.isabs(rel_or_abs):
            # Legacy absolute path stored before this version
            return rel_or_abs
        return os.path.normpath(os.path.join(self.output_dir, rel_or_abs))

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename     TEXT    NOT NULL,
                    filepath     TEXT    NOT NULL UNIQUE,
                    processed    INTEGER NOT NULL DEFAULT 0,
                    face_count   INTEGER NOT NULL DEFAULT 0,
                    orig_width   INTEGER,
                    orig_height  INTEGER,
                    base_path    TEXT,
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS faces (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id       INTEGER NOT NULL REFERENCES images(id),
                    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                    embedding      BLOB    NOT NULL,
                    det_score      REAL    NOT NULL DEFAULT 0.0,
                    thumbnail_path TEXT,
                    person_id      INTEGER REFERENCES persons(id)
                );

                CREATE TABLE IF NOT EXISTS persons (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_id      TEXT    UNIQUE NOT NULL,
                    name            TEXT,
                    surname         TEXT,
                    face_count      INTEGER NOT NULL DEFAULT 0,
                    representative  TEXT,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
                CREATE INDEX IF NOT EXISTS idx_faces_image  ON faces(image_id);
            """)

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    def upsert_image(self, filename: str, filepath: str) -> int:
        filepath = os.path.abspath(filepath)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO images (filename, filepath) VALUES (?, ?)",
                (filename, filepath),
            )
            row = conn.execute(
                "SELECT id FROM images WHERE filepath = ?", (filepath,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to upsert image: {filepath}")
            return int(row["id"])

    def mark_processed(
        self,
        image_id: int,
        face_count: int,
        orig_width: int,
        orig_height: int,
        base_path: str,
    ):
        with self._conn() as conn:
            conn.execute(
                """UPDATE images
                   SET processed=1, face_count=?, orig_width=?, orig_height=?, base_path=?
                   WHERE id=?""",
                (face_count, orig_width or 0, orig_height or 0,
                 self._rel(base_path), image_id),
            )

    def get_unprocessed(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, filename, filepath FROM images WHERE processed=0 ORDER BY filename"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_images(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, filename, filepath, processed, face_count,
                          orig_width, orig_height, base_path
                   FROM images ORDER BY filename"""
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["base_path"] = self._abs(d["base_path"] or "")
            result.append(d)
        return result

    def count_images(self) -> Tuple[int, int]:
        """Returns (total, processed)."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            done  = conn.execute(
                "SELECT COUNT(*) FROM images WHERE processed=1"
            ).fetchone()[0]
        return int(total), int(done)

    # ------------------------------------------------------------------
    # faces
    # ------------------------------------------------------------------

    def insert_face(
        self,
        image_id: int,
        bbox: Tuple,
        embedding: np.ndarray,
        det_score: float,
        thumbnail_path: str,
    ) -> int:
        blob = embedding.astype(np.float32).tobytes()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO faces
                   (image_id, x1, y1, x2, y2, embedding, det_score, thumbnail_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    int(bbox[0]), int(bbox[1]),
                    int(bbox[2]), int(bbox[3]),
                    blob,
                    float(det_score),
                    self._rel(thumbnail_path),
                ),
            )
            fid = cur.lastrowid
            if not fid:
                raise RuntimeError("insert_face: lastrowid is None")
            return int(fid)

    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM faces ORDER BY id"
            ).fetchall()

        result = []
        for r in rows:
            try:
                blob = r["embedding"]
                if blob is None or len(blob) == 0:
                    logger.warning("Face id=%s has empty embedding – skipping", r["id"])
                    continue
                emb = np.frombuffer(blob, dtype=np.float32).copy()
                if emb.shape[0] == 0:
                    logger.warning("Face id=%s deserialized to zero-length embedding", r["id"])
                    continue
                result.append((int(r["id"]), emb))
            except (ValueError, TypeError) as exc:
                logger.warning("Face id=%s corrupted embedding: %s – skipping", r["id"], exc)
        return result

    def assign_persons(self, pairs: List[Tuple[int, int]]):
        """pairs: [(face_id, person_id), ...]"""
        if not pairs:
            return
        with self._conn() as conn:
            conn.executemany(
                "UPDATE faces SET person_id=? WHERE id=?",
                [(pid, fid) for fid, pid in pairs],
            )

    def get_faces_for_person(self, person_id: int) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT f.id, f.thumbnail_path, f.det_score,
                          f.x1, f.y1, f.x2, f.y2,
                          i.filename, i.filepath, i.base_path,
                          i.orig_width, i.orig_height
                   FROM faces f JOIN images i ON f.image_id = i.id
                   WHERE f.person_id = ?
                   ORDER BY f.det_score DESC""",
                (person_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["thumbnail_path"] = self._abs(d["thumbnail_path"] or "")
            d["base_path"]      = self._abs(d["base_path"] or "")
            result.append(d)
        return result

    def get_faces_for_image(self, image_id: int) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT f.id, f.x1, f.y1, f.x2, f.y2,
                          f.det_score, f.person_id,
                          p.display_id, p.name, p.surname
                   FROM faces f
                   LEFT JOIN persons p ON f.person_id = p.id
                   WHERE f.image_id = ?""",
                (image_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def reset_clustering(self):
        with self._conn() as conn:
            conn.execute("UPDATE faces SET person_id = NULL")
            conn.execute("DELETE FROM persons")

    def merge_persons(self, keep_id: int, remove_id: int):
        """Reassign all faces from remove_id to keep_id, then delete remove_id."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE faces SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )
            conn.execute("DELETE FROM persons WHERE id=?", (remove_id,))
        # Update stats for the surviving person
        faces = self.get_faces_for_person(keep_id)
        rep   = faces[0]["thumbnail_path"] if faces else None
        self.update_person_meta(keep_id, len(faces), rep or "")

    def delete_person(self, person_id: int):
        """Unassign all faces from this person and delete the person record."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE faces SET person_id = NULL WHERE person_id = ?",
                (person_id,),
            )
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    # ------------------------------------------------------------------
    # persons
    # ------------------------------------------------------------------

    def create_person(self, display_id: str) -> int:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO persons (display_id) VALUES (?)",
                (display_id,),
            )
            row = conn.execute(
                "SELECT id FROM persons WHERE display_id = ?", (display_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"create_person: cannot find '{display_id}'")
            return int(row["id"])

    def update_person_meta(self, person_id: int, face_count: int, representative: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE persons SET face_count=?, representative=? WHERE id=?",
                (face_count, self._rel(representative) if representative else None, person_id),
            )

    def update_person_name(self, person_id: int, name: str, surname: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE persons SET name=?, surname=? WHERE id=?",
                (name or None, surname or None, person_id),
            )

    def get_all_persons(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, display_id, name, surname, face_count, representative
                   FROM persons ORDER BY face_count DESC, display_id"""
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["representative"] = self._abs(d["representative"] or "")
            result.append(d)
        return result

    def get_person(self, person_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, display_id, name, surname, face_count, representative
                   FROM persons WHERE id=?""",
                (person_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["representative"] = self._abs(d["representative"] or "")
        return d

    def get_stats(self) -> Dict:
        with self._conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            done    = conn.execute(
                "SELECT COUNT(*) FROM images WHERE processed=1"
            ).fetchone()[0]
            faces   = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            named   = conn.execute(
                "SELECT COUNT(*) FROM persons WHERE name IS NOT NULL AND name != ''"
            ).fetchone()[0]
            persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            unassigned = conn.execute(
                "SELECT COUNT(*) FROM faces WHERE person_id IS NULL"
            ).fetchone()[0]
        db_size_mb = os.path.getsize(self.db_path) / 1_048_576 if os.path.exists(self.db_path) else 0
        return {
            "total_images":     int(total),
            "processed_images": int(done),
            "total_faces":      int(faces),
            "total_persons":    int(persons),
            "named_persons":    int(named),
            "unassigned_faces": int(unassigned),
            "db_size_mb":       round(db_size_mb, 2),
        }

    def get_top_persons(self, n: int = 10) -> List[Dict]:
        """Return top-n persons by face count for statistics display."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, display_id, name, surname, face_count
                   FROM persons
                   ORDER BY face_count DESC
                   LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_images_with_most_faces(self, n: int = 10) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT filename, face_count
                   FROM images WHERE processed=1
                   ORDER BY face_count DESC
                   LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_person_image_counts(self) -> List[Dict]:
        """For each person, count the number of distinct images they appear in."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT p.id, p.display_id, p.name, p.surname,
                          COUNT(DISTINCT f.image_id) AS image_count,
                          p.face_count
                   FROM persons p
                   JOIN faces f ON f.person_id = p.id
                   GROUP BY p.id
                   ORDER BY image_count DESC"""
            ).fetchall()
        return [dict(r) for r in rows]
