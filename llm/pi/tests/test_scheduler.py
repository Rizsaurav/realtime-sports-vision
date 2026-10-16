"""Hidden acceptance tests for the scheduler library (spec: llm/pi/specs/scheduler.md)."""
import os
import sys

import pytest

sys.path.insert(0, os.environ.get("APP_DIR", "."))
import scheduler  # noqa: E402
from scheduler import CycleError, UnknownTaskError, critical_path, plan, simulate  # noqa: E402

DIAMOND = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
DUR = {"a": 2, "b": 3, "c": 1, "d": 4}


def assert_cycle(exc, graph):
    cyc = exc.cycle
    assert len(cyc) >= 2 and cyc[0] == cyc[-1]
    for x, y in zip(cyc, cyc[1:]):
        assert y in graph[x], f"{x} does not depend on {y}"


def test_exception_hierarchy():
    assert issubclass(CycleError, ValueError) and issubclass(UnknownTaskError, KeyError)


def test_plan_diamond():
    assert plan(DIAMOND) == [["a"], ["b", "c"], ["d"]]


def test_plan_independent_tasks_sorted():
    assert plan({"b": [], "a": [], "c": []}) == [["a", "b", "c"]]


def test_plan_levels_use_longest_dependency_chain():
    g = {"x": [], "y": ["x"], "z": ["x", "y"], "w": []}
    assert plan(g) == [["w", "x"], ["y"], ["z"]]


def test_plan_empty_and_single():
    assert plan({}) == []
    assert plan({"only": []}) == [["only"]]


def test_plan_does_not_modify_input():
    g = {"a": [], "b": ["a"]}
    plan(g)
    assert g == {"a": [], "b": ["a"]}


def test_unknown_dependency():
    with pytest.raises(UnknownTaskError) as e:
        plan({"a": ["ghost"]})
    assert "ghost" in str(e.value)


def test_cycle_detected_with_valid_cycle_path():
    g = {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}
    with pytest.raises(CycleError) as e:
        plan(g)
    assert_cycle(e.value, g)


def test_self_dependency_is_a_cycle():
    with pytest.raises(CycleError) as e:
        plan({"a": ["a"]})
    assert e.value.cycle == ["a", "a"]


def test_cycle_in_other_functions():
    g = {"a": ["b"], "b": ["a"]}
    d = {"a": 1, "b": 1}
    with pytest.raises(CycleError):
        critical_path(g, d)
    with pytest.raises(CycleError):
        simulate(g, d, 2)


def test_critical_path_diamond():
    total, path = critical_path(DIAMOND, DUR)
    assert total == 9.0 and path == ["a", "b", "d"] and isinstance(total, float)


def test_critical_path_tie_breaks_by_smallest_list():
    total, path = critical_path(DIAMOND, {"a": 1, "b": 2, "c": 2, "d": 1})
    assert total == 4.0 and path == ["a", "b", "d"]


def test_critical_path_picks_heavier_branch():
    total, path = critical_path(DIAMOND, {"a": 1, "b": 1, "c": 5, "d": 1})
    assert total == 7.0 and path == ["a", "c", "d"]


def test_critical_path_disconnected_and_empty():
    assert critical_path({"p": [], "q": []}, {"p": 5, "q": 5}) == (5.0, ["p"])
    assert critical_path({"p": [], "q": []}, {"p": 5, "q": 6}) == (6.0, ["q"])
    assert critical_path({}, {}) == (0.0, [])


def test_critical_path_zero_durations():
    # every chain totals 0; ["a"] is the smallest list of names, which is what the spec asks for
    total, path = critical_path({"a": [], "b": ["a"]}, {"a": 0, "b": 0})
    assert total == 0.0 and path == ["a"]


def test_critical_path_validation():
    with pytest.raises(ValueError):
        critical_path({"a": []}, {"a": -1})
    with pytest.raises(ValueError):
        critical_path({"a": [], "b": []}, {"a": 1})


def test_simulate_one_worker():
    span, sched = simulate(DIAMOND, DUR, 1)
    assert span == 10 and sched == {"a": (0, 2), "b": (2, 5), "c": (5, 6), "d": (6, 10)}


@pytest.mark.parametrize("workers", [2, 3, 10])
def test_simulate_parallel_workers(workers):
    span, sched = simulate(DIAMOND, DUR, workers)
    assert span == 9 and sched == {"a": (0, 2), "b": (2, 5), "c": (2, 3), "d": (5, 9)}


def test_simulate_independent_tasks_two_workers():
    g = {f"t{i}": [] for i in range(1, 5)}
    span, sched = simulate(g, {k: 3 for k in g}, 2)
    assert span == 6 and sched == {"t1": (0, 3), "t2": (0, 3), "t3": (3, 6), "t4": (3, 6)}


def test_simulate_smallest_name_starts_first():
    span, sched = simulate({"b": [], "a": []}, {"a": 1, "b": 1}, 1)
    assert sched == {"a": (0, 1), "b": (1, 2)} and span == 2


def test_simulate_waiting_task_gets_freed_worker():
    g = {"a": [], "b": [], "c": []}
    span, sched = simulate(g, {"a": 2, "b": 2, "c": 1}, 2)
    assert span == 3 and sched == {"a": (0, 2), "b": (0, 2), "c": (2, 3)}


def test_simulate_dependencies_release_tasks_when_finished():
    g = {"a": [], "b": [], "c": ["a"], "d": ["b"]}
    span, sched = simulate(g, {"a": 1, "b": 3, "c": 3, "d": 1}, 2)
    assert span == 4 and sched == {"a": (0, 1), "b": (0, 3), "c": (1, 4), "d": (3, 4)}


def test_simulate_zero_duration_chain():
    span, sched = simulate({"a": [], "b": ["a"]}, {"a": 0, "b": 2}, 1)
    assert span == 2 and sched == {"a": (0, 0), "b": (0, 2)}


def test_simulate_empty_and_validation():
    assert simulate({}, {}, 3) == (0.0, {})
    with pytest.raises(ValueError):
        simulate(DIAMOND, DUR, 0)
    with pytest.raises(UnknownTaskError):
        simulate({"a": ["zz"]}, {"a": 1}, 1)


def test_module_exposes_the_public_names():
    for name in ("plan", "critical_path", "simulate", "CycleError", "UnknownTaskError"):
        assert hasattr(scheduler, name)
