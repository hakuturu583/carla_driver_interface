# SPDX-License-Identifier: Apache-2.0
"""The order in which a rollout tears its scenario down.

Ordinarily an ordering like this would not be worth a test.  This one is,
because getting it wrong does not raise -- it aborts the process from a thread
Python cannot see, after the rollout has already succeeded, and the only
evidence is a ``SIGABRT`` and a C++ message on stderr.
"""

from __future__ import annotations

from types import SimpleNamespace

from carla_driver_interface.runtime.carla_world import CarlaWorldAdapter


class _Recorder:
    """Stands in for the CARLA client, recording what is called and when."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def actor(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            destroy=lambda: self.events.append(f"destroy:{name}"),
            stop=lambda: self.events.append(f"stop:{name}"),
        )

    def traffic_manager(self) -> SimpleNamespace:
        def set_synchronous_mode(enabled: bool) -> None:
            self.events.append(f"traffic_manager_sync:{enabled}")

        return SimpleNamespace(set_synchronous_mode=set_synchronous_mode)


def build_adapter(recorder: _Recorder) -> CarlaWorldAdapter:
    """A bare adapter with only the fields ``close`` touches.

    Constructed without ``__init__`` because the real one needs a CARLA
    server, and the property under test is pure bookkeeping.
    """
    adapter = object.__new__(CarlaWorldAdapter)
    adapter._sensors = [recorder.actor("camera")]
    adapter._background = [recorder.actor("bg0"), recorder.actor("bg1")]
    adapter._ego = recorder.actor("ego")
    adapter._traffic_manager = recorder.traffic_manager()
    # No client, so `close` takes the per-actor path and the recorder sees
    # each destruction individually.
    adapter._client = None
    adapter._world = None
    adapter._original_settings = None
    return adapter


def test_the_traffic_manager_stands_down_before_its_vehicles_are_destroyed():
    """Otherwise it keeps driving actors that no longer exist.

    The traffic manager runs a thread inside the CARLA client and, while in
    synchronous mode, issues commands every tick for every vehicle registered
    to it. Destroying those vehicles first makes it operate on destroyed
    actors; the C++ exception surfaces on its own thread, where no Python
    handler can catch it, and the process aborts.
    """
    recorder = _Recorder()
    build_adapter(recorder).close()

    sync_off = recorder.events.index("traffic_manager_sync:False")
    first_destroy = min(
        index for index, event in enumerate(recorder.events) if event.startswith("destroy:")
    )
    assert sync_off < first_destroy


def test_every_actor_is_destroyed():
    recorder = _Recorder()
    build_adapter(recorder).close()

    destroyed = {
        event.split(":", 1)[1] for event in recorder.events if event.startswith("destroy:")
    }
    assert destroyed == {"camera", "bg0", "bg1", "ego"}


def test_sensors_are_stopped_before_they_are_destroyed():
    """A sensor still streaming into a destroyed callback is the same hazard."""
    recorder = _Recorder()
    build_adapter(recorder).close()
    assert recorder.events.index("stop:camera") < recorder.events.index("destroy:camera")


def test_close_leaves_no_actor_references_behind():
    recorder = _Recorder()
    adapter = build_adapter(recorder)
    adapter.close()
    assert adapter._background == []
    assert adapter._ego is None
    assert adapter._sensors == []


def test_close_is_idempotent():
    """The runtime closes in a `finally`, so a second call must be harmless."""
    recorder = _Recorder()
    adapter = build_adapter(recorder)
    adapter.close()
    adapter.close()


def test_a_failing_actor_teardown_does_not_stop_the_rest():
    """One already-gone actor must not strand the others."""
    recorder = _Recorder()
    adapter = build_adapter(recorder)

    def explode() -> None:
        raise RuntimeError("actor already destroyed")

    adapter._background[0] = SimpleNamespace(destroy=explode)
    adapter.close()

    destroyed = {
        event.split(":", 1)[1] for event in recorder.events if event.startswith("destroy:")
    }
    assert {"bg1", "ego"} <= destroyed
