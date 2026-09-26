"""ActivityTracker unit tests."""

from __future__ import annotations

from app.activity import ActivityTracker


def test_activity_start_update_finish_snapshot() -> None:
    tracker = ActivityTracker(recent_limit=5)
    tracker.start(
        "abc123",
        src_w=300,
        src_h=300,
        pads="64,64,64,64",
        out_size="428x428",
        quality_gate=False,
    )
    snap = tracker.snapshot()
    assert len(snap["current"]) == 1
    assert snap["current"][0]["phase"] == "queued"
    assert snap["recent"] == []

    tracker.update(
        "abc123",
        phase="waiting",
        prompt_id="pid-1",
        seed=42,
        positive_prompt="extend",
        negative_prompt="text",
        comfy_image="album_cover.png",
    )
    cur = tracker.snapshot()["current"][0]
    assert cur["phase"] == "waiting"
    assert cur["prompt_id"] == "pid-1"
    assert cur["seed"] == 42
    assert cur["comfy_image"] == "album_cover.png"

    tracker.finish("abc123", phase="done")
    snap = tracker.snapshot()
    assert snap["current"] == []
    assert len(snap["recent"]) == 1
    assert snap["recent"][0]["phase"] == "done"
    assert snap["recent"][0]["prompt_id"] == "pid-1"


def test_activity_finish_unknown_is_noop() -> None:
    tracker = ActivityTracker()
    tracker.finish("missing", phase="error", error="nope")
    assert tracker.snapshot() == {"current": [], "recent": []}
