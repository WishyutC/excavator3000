"""Read and summarize the live training CSV without interrupting its writer."""

from collections import Counter
import csv
from datetime import datetime, timezone
from pathlib import Path
import threading


class TrainingDataReader:
    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self._lock = threading.Lock()
        self._mtime_ns = None
        self._rows = []

    @staticmethod
    def _number(row, key, conversion, default=0):
        try:
            return conversion(row.get(key, default))
        except (TypeError, ValueError):
            return default

    def rows(self):
        if not self.path.exists():
            return []
        mtime_ns = self.path.stat().st_mtime_ns
        with self._lock:
            if mtime_ns == self._mtime_ns:
                return list(self._rows)
            parsed = []
            try:
                with self.path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if not row.get("episode"):
                            continue
                        parsed.append({
                            "episode": self._number(row, "episode", int),
                            "steps": self._number(row, "steps", int),
                            "reward": self._number(row, "total_reward", float),
                            "reason": row.get("termination_reason", "unknown"),
                            "success": row.get("success", "").lower() == "true",
                            "epsilon": self._number(row, "epsilon", float),
                            "buffer_size": self._number(row, "buffer_size", int),
                            "training_steps": self._number(row, "training_steps", int),
                            "loss": self._number(row, "loss", float, None)
                        })
            except (OSError, csv.Error):
                return list(self._rows)
            self._rows = parsed
            self._mtime_ns = mtime_ns
            return list(self._rows)

    @staticmethod
    def _window_summary(rows):
        if not rows:
            return {
                "count": 0, "average_reward": None, "best_reward": None,
                "success_rate": 0.0, "collision_rate": 0.0,
                "timeout_rate": 0.0, "reasons": {}
            }
        reasons = Counter(row["reason"] for row in rows)
        rewards = [row["reward"] for row in rows]
        count = len(rows)
        return {
            "count": count,
            "average_reward": sum(rewards) / count,
            "best_reward": max(rewards),
            "success_rate": 100.0 * reasons["goal_reached"] / count,
            "collision_rate": 100.0 * reasons["collision"] / count,
            "timeout_rate": 100.0 * reasons["timeout"] / count,
            "reasons": dict(reasons)
        }

    @staticmethod
    def _downsample(rows, maximum=600):
        if len(rows) <= maximum:
            return rows
        stride = max(1, len(rows) // maximum)
        sampled = rows[::stride]
        if sampled[-1] is not rows[-1]:
            sampled.append(rows[-1])
        return sampled[-maximum:]

    def snapshot(self):
        rows = self.rows()
        history_source = rows[-2_000:]
        rolling_rewards = []
        running = []
        history = []
        for row in history_source:
            running.append(row["reward"])
            if len(running) > 20:
                running.pop(0)
            item = dict(row)
            item["rolling_reward_20"] = sum(running) / len(running)
            history.append(item)

        modified = None
        stale_seconds = None
        if self.path.exists():
            timestamp = self.path.stat().st_mtime
            modified = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            stale_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - timestamp)

        latest = rows[-1] if rows else None
        recent_20 = self._window_summary(rows[-20:])
        previous_20 = self._window_summary(rows[-40:-20])
        trend = None
        if recent_20["average_reward"] is not None and previous_20["average_reward"] is not None:
            trend = recent_20["average_reward"] - previous_20["average_reward"]

        return {
            "has_data": bool(rows),
            "latest": latest,
            "windows": {
                "20": recent_20,
                "100": self._window_summary(rows[-100:])
            },
            "reward_trend_20": trend,
            "history": self._downsample(history),
            "recent_rows": list(reversed(rows[-12:])),
            "csv_updated_at": modified,
            "csv_stale_seconds": stale_seconds,
            "total_rows": len(rows)
        }
