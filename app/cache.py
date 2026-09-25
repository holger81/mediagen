"""Frequency-aware content-hash disk cache for generated media."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.constants import OUTPAINT_CACHE_VERSION
from app.image_id import content_hash as image_content_hash
from app.layout import OutpaintLayout


@dataclass(frozen=True)
class CacheHit:
    hash: str
    path: Path
    source: str  # flux | local
    hits: int


@dataclass(frozen=True)
class CacheEntryInfo:
    hash: str
    source: str
    hits: int
    created_at: float
    last_access: float
    size_bytes: int
    done: bool
    flux: bool
    pads: str | None


class MediaCache:
    """Disk JPEG cache keyed by canonical image identity with two-tier eviction."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_items: int = 1000,
        cache_version: str = OUTPAINT_CACHE_VERSION,
        media_type: str = "image",
        operation: str = "outpaint",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_items = max(1, max_items)
        self.cache_version = cache_version
        self.media_type = media_type
        self.operation = operation
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.cache_dir / "index.sqlite3"
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS entries (
                        hash TEXT PRIMARY KEY,
                        media_type TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        source TEXT NOT NULL,
                        hits INTEGER NOT NULL DEFAULT 1,
                        last_access REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_hits_access ON entries(hits, last_access)"
            )

    @staticmethod
    def content_hash(
        source_bytes: bytes,
        cache_version: str = OUTPAINT_CACHE_VERSION,
        *,
        layout: OutpaintLayout | None = None,
    ) -> str:
        return image_content_hash(source_bytes, cache_version, layout=layout)

    def hash_for(
        self,
        source_bytes: bytes,
        *,
        layout: OutpaintLayout | None = None,
    ) -> str:
        return self.content_hash(source_bytes, self.cache_version, layout=layout)

    def file_for(self, content_hash: str) -> Path:
        return self.cache_dir / f"{content_hash}.jpg"

    def done_marker_for(self, content_hash: str) -> Path:
        return self.cache_dir / f"{content_hash}.done"

    def pads_marker_for(self, content_hash: str) -> Path:
        return self.cache_dir / f"{content_hash}.pads"

    def write_layout(self, content_hash: str, layout: OutpaintLayout) -> None:
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.pads_marker_for(content_hash).write_text(layout.header_pad(), encoding="utf-8")

    def read_layout(self, content_hash: str) -> OutpaintLayout | None:
        path = self.pads_marker_for(content_hash)
        if not path.is_file():
            return None
        try:
            parts = path.read_text(encoding="utf-8").strip().split(",")
            if len(parts) != 4:
                return None
            return OutpaintLayout(
                pad_left=int(parts[0]),
                pad_top=int(parts[1]),
                pad_right=int(parts[2]),
                pad_bottom=int(parts[3]),
            )
        except (OSError, ValueError):
            return None

    def mark_done(self, content_hash: str) -> None:
        """Mark a generation attempt finished (Flux or local fallback)."""
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.done_marker_for(content_hash).touch()

    def is_done(self, content_hash: str) -> bool:
        return self.done_marker_for(content_hash).is_file()

    def get(
        self,
        source_bytes: bytes,
        *,
        touch: bool = True,
        layout: OutpaintLayout | None = None,
    ) -> CacheHit | None:
        content_hash = self.hash_for(source_bytes, layout=layout)
        return self.get_by_hash(content_hash, touch=touch)

    def get_by_hash(self, content_hash: str, *, touch: bool = True) -> CacheHit | None:
        path = self.file_for(content_hash)
        with self._lock:
            if not path.is_file() or path.stat().st_size <= 0:
                return None
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT hash, source, hits FROM entries WHERE hash = ?",
                    (content_hash,),
                ).fetchone()
                if row is None:
                    # Orphan file — register as local with 1 hit.
                    now = time.time()
                    conn.execute(
                        """
                        INSERT INTO entries
                        (hash, media_type, operation, source, hits, last_access, created_at)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            content_hash,
                            self.media_type,
                            self.operation,
                            "local",
                            now,
                            now,
                        ),
                    )
                    return CacheHit(hash=content_hash, path=path, source="local", hits=1)
                hits = int(row["hits"])
                source = str(row["source"])
                if touch:
                    hits += 1
                    conn.execute(
                        "UPDATE entries SET hits = ?, last_access = ? WHERE hash = ?",
                        (hits, time.time(), content_hash),
                    )
                return CacheHit(hash=content_hash, path=path, source=source, hits=hits)

    def put(
        self,
        source_bytes: bytes,
        result_bytes: bytes,
        *,
        source: str,
        layout: OutpaintLayout | None = None,
    ) -> CacheHit | None:
        if not result_bytes:
            return None
        content_hash = self.hash_for(source_bytes, layout=layout)
        return self.put_by_hash(content_hash, result_bytes, source=source)

    def put_by_hash(
        self,
        content_hash: str,
        result_bytes: bytes,
        *,
        source: str,
    ) -> CacheHit | None:
        if not result_bytes:
            return None
        path = self.file_for(content_hash)
        tmp = self.cache_dir / f"{content_hash}.tmp"
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(result_bytes)
            tmp.replace(path)
            now = time.time()
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT hits FROM entries WHERE hash = ?", (content_hash,)
                ).fetchone()
                if existing is None:
                    hits = 1
                    conn.execute(
                        """
                        INSERT INTO entries
                        (hash, media_type, operation, source, hits, last_access, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            content_hash,
                            self.media_type,
                            self.operation,
                            source,
                            hits,
                            now,
                            now,
                        ),
                    )
                else:
                    hits = int(existing["hits"]) + 1
                    conn.execute(
                        """
                        UPDATE entries
                        SET source = ?, hits = ?, last_access = ?
                        WHERE hash = ?
                        """,
                        (source, hits, now, content_hash),
                    )
                self._enforce_limits(conn)
            return CacheHit(hash=content_hash, path=path, source=source, hits=hits)

    def invalidate(self, content_hash: str) -> bool:
        """Remove a cache entry and markers. Returns True if a file was removed."""
        path = self.file_for(content_hash)
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM entries WHERE hash = ?", (content_hash,))
            removed = path.is_file()
            path.unlink(missing_ok=True)
            (self.cache_dir / f"{content_hash}.flux").unlink(missing_ok=True)
            self.done_marker_for(content_hash).unlink(missing_ok=True)
            self.pads_marker_for(content_hash).unlink(missing_ok=True)
            return removed

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()
            return int(row["n"]) if row else 0

    def list_entries(self, *, limit: int = 200, offset: int = 0) -> list[CacheEntryInfo]:
        """List cache entries newest-access first (for admin UI). Does not bump hits."""
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT hash, source, hits, last_access, created_at
                FROM entries
                ORDER BY last_access DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        out: list[CacheEntryInfo] = []
        for row in rows:
            content_hash = str(row["hash"])
            path = self.file_for(content_hash)
            try:
                size = path.stat().st_size if path.is_file() else 0
            except OSError:
                size = 0
            layout = self.read_layout(content_hash)
            out.append(
                CacheEntryInfo(
                    hash=content_hash,
                    source=str(row["source"]),
                    hits=int(row["hits"]),
                    created_at=float(row["created_at"]),
                    last_access=float(row["last_access"]),
                    size_bytes=size,
                    done=self.is_done(content_hash),
                    flux=(self.cache_dir / f"{content_hash}.flux").is_file(),
                    pads=layout.header_pad() if layout else None,
                )
            )
        return out

    def _enforce_limits(self, conn: sqlite3.Connection) -> None:
        while True:
            row = conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()
            total = int(row["n"]) if row else 0
            if total <= self.max_items:
                return
            # Probation (hits == 1) first, oldest last_access; then protected.
            victim = conn.execute(
                """
                SELECT hash FROM entries
                WHERE hits = 1
                ORDER BY last_access ASC
                LIMIT 1
                """
            ).fetchone()
            if victim is None:
                victim = conn.execute(
                    """
                    SELECT hash FROM entries
                    ORDER BY hits ASC, last_access ASC
                    LIMIT 1
                    """
                ).fetchone()
            if victim is None:
                return
            victim_hash = str(victim["hash"])
            conn.execute("DELETE FROM entries WHERE hash = ?", (victim_hash,))
            path = self.file_for(victim_hash)
            path.unlink(missing_ok=True)
            (self.cache_dir / f"{victim_hash}.flux").unlink(missing_ok=True)
            self.done_marker_for(victim_hash).unlink(missing_ok=True)
            self.pads_marker_for(victim_hash).unlink(missing_ok=True)
