"""Persistence helpers for post-run Feedback Plane evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from rosclaw.feedback.contracts import FeedbackReceipt


def write_feedback_receipt(path: Path, receipt: FeedbackReceipt) -> Path:
    """Atomically write a receipt outside the synchronous control path."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt.to_dict(), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination
