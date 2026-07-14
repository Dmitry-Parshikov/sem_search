"""Structured query logging (Ф4.2): query text, response time, search mode,
index version, and params -- one JSON line per search request.

Deliberately not routed through `structlog`'s (stderr-oriented) default
configuration: an audit log needs to land in a specific FILE
(`settings.admin.query_log_path`, a `.jsonl` path per the config schema),
independent of whatever operational logging setup the rest of the app uses.
`QueryLogger` is a small, directly-testable JSONL appender instead.

Opens the file in append mode per call (simplicity over throughput -- this is
a thesis prototype, not a high-throughput system); append-mode writes of a
single line are safe for sequential/concurrent-but-uncontended calls since
each `.log()` call performs one `write()` of a complete, newline-terminated
line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class QueryLogger:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def log(self, entry: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
