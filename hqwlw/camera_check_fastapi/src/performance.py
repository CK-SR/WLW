"""Utilities for collecting and exposing runtime performance metrics.

The module keeps lightweight, in-memory statistics for the processing
pipeline.  Each stage (e.g. decoding, analysis, MinIO upload) is tracked
per stream as well as globally.  The statistics are primarily intended to
feed observability dashboards – the FastAPI layer exposes them via a JSON
endpoint so they can be visualised by an external UI (Grafana, ECharts,
etc.).

The implementation purposely avoids heavy dependencies (Prometheus client,
NumPy, …) so the monitoring can run inside worker processes without any
additional infrastructure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Deque, Dict, Iterable, Mapping, MutableMapping, Optional
from datetime import datetime, timezone


def _now_iso_utc() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _percentile(samples: Iterable[float], pct: float) -> Optional[float]:
    values = sorted(samples)
    if not values:
        return None

    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]

    k = (len(values) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


@dataclass
class StageStatistics:
    """Rolling window statistics for a specific processing stage."""

    history_size: int
    total: float = 0.0
    count: int = 0
    maximum: float = 0.0
    minimum: float = float("inf")
    history: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.history_size)

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        self.history.append(value)

    @property
    def average(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count

    def as_dict(self) -> Dict[str, Optional[float]]:
        samples = list(self.history)
        return {
            "count": self.count,
            "avg": None if self.average is None else round(self.average, 3),
            "p95": None if not samples else round(_percentile(samples, 95), 3),
            "max": None if self.count == 0 else round(self.maximum, 3),
            "min": None if self.count == 0 else round(self.minimum, 3),
            "last": None if not samples else round(samples[-1], 3),
            "history": [round(s, 3) for s in samples],
        }


class PerformanceMonitor:
    """Central registry used to collect per-stream pipeline metrics."""

    def __init__(self, history_size: int = 120, frame_history: int = 60):
        self._history_size = history_size
        self._frame_history = frame_history
        self._streams: MutableMapping[str, MutableMapping[str, StageStatistics]] = defaultdict(self._new_stage_map)
        self._global: MutableMapping[str, StageStatistics] = defaultdict(self._new_stage)
        self._frame_records: MutableMapping[str, Deque[Mapping[str, object]]] = defaultdict(
            lambda: deque(maxlen=self._frame_history)
        )
        self._lock = RLock()

    def _new_stage(self) -> StageStatistics:
        return StageStatistics(history_size=self._history_size)

    def _new_stage_map(self) -> MutableMapping[str, StageStatistics]:
        return defaultdict(self._new_stage)

    def configure(self, *, history_size: Optional[int] = None, frame_history: Optional[int] = None) -> None:
        """Adjust the rolling window sizes used for metrics collection."""

        with self._lock:
            if history_size and history_size != self._history_size:
                self._history_size = history_size
                for stage_map in self._streams.values():
                    for stats in stage_map.values():
                        stats.history_size = history_size
                        stats.history = deque(stats.history, maxlen=history_size)
                for stats in self._global.values():
                    stats.history_size = history_size
                    stats.history = deque(stats.history, maxlen=history_size)
            if frame_history and frame_history != self._frame_history:
                self._frame_history = frame_history
                for stream, records in list(self._frame_records.items()):
                    self._frame_records[stream] = deque(records, maxlen=self._frame_history)

    def record_stage(self, stream: str, stage: str, value_ms: Optional[float]) -> None:
        if value_ms is None:
            return
        with self._lock:
            stage_stats = self._streams[stream][stage]
            stage_stats.record(value_ms)
            self._global[stage].record(value_ms)

    def record_frame(self, stream: str, timings_ms: Mapping[str, Optional[float]], **extra: object) -> None:
        """Record a set of stage timings for a frame.

        The *timings_ms* mapping may contain ``None`` values for stages that were
        not available.  Those values are ignored for statistical purposes but the
        entire payload is retained in the per-frame history deque so that callers
        can visualise the timeline later.
        """

        payload = {
            "recorded_at": _now_iso_utc(),
            "timings": {k: v for k, v in timings_ms.items() if v is not None},
        }
        if extra:
            payload.update(extra)

        with self._lock:
            for stage, value in timings_ms.items():
                if value is None:
                    continue
                stage_stats = self._streams[stream][stage]
                stage_stats.record(float(value))
                self._global[stage].record(float(value))
            self._frame_records[stream].append(payload)

    def reset(self) -> None:
        with self._lock:
            self._streams.clear()
            self._global.clear()
            self._frame_records.clear()

    def snapshot(self, stream: Optional[str] = None) -> Dict[str, object]:
        with self._lock:
            if stream is not None:
                stream_stats = {
                    stage: stats.as_dict()
                    for stage, stats in self._streams.get(stream, {}).items()
                }
                return {
                    "stream": stream,
                    "updated_at": _now_iso_utc(),
                    "stages": stream_stats,
                    "frames": list(self._frame_records.get(stream, [])),
                }

            all_streams = {
                name: {stage: stats.as_dict() for stage, stats in stages.items()}
                for name, stages in self._streams.items()
            }
            return {
                "updated_at": _now_iso_utc(),
                "streams": all_streams,
                "global": {stage: stats.as_dict() for stage, stats in self._global.items()},
            }


# Global singleton used by the application
PERFORMANCE_MONITOR = PerformanceMonitor()

