# Task scheduler library

Please write a Python 3 module `scheduler.py` (standard library only) for planning and simulating tasks with
dependencies. A task graph is a dict that maps each task name (a string) to the list of names it depends on (those
must finish before it can start): `{"build": [], "test": ["build"], "ship": ["test", "build"]}`.

## Exceptions

- `CycleError(ValueError)`: raised for a graph with a dependency cycle. Its attribute `cycle` is a list of names
  `[n0, n1, ..., n0]` that starts and ends with the same name, where each name depends on the next one. A task that
  depends on itself gives `["a", "a"]`.
- `UnknownTaskError(KeyError)`: raised when a task depends on a name that is not a key of the graph. The message
  contains the unknown name.

## Functions (all validate the graph and raise the exceptions above)

1. `plan(tasks) -> list[list[str]]` returns the tasks grouped into levels that could run in parallel. The level of a
   task is 0 if it has no dependencies, otherwise 1 plus the highest level among its dependencies. Each level is
   sorted alphabetically and the levels are in increasing order. An empty graph gives `[]`.
2. `critical_path(tasks, durations) -> tuple[float, list[str]]` where `durations` maps every task name to a number of
   seconds. It returns the total duration of the most expensive chain of dependent tasks (the sum of the durations of the
   tasks on it) and that chain, ordered from the first task to the last. If several chains have the same total, return
   the one that is smallest when compared as a list of names. A negative duration, or a task without a duration, raises
   `ValueError`. An empty graph gives `(0.0, [])`.
3. `simulate(tasks, durations, workers) -> tuple[float, dict[str, tuple[float, float]]]` runs the tasks on `workers`
   identical workers (an integer of at least 1, otherwise `ValueError`) and returns the makespan (the time when the
   last task ends, `0.0` for an empty graph) and a dict mapping each task to its `(start, end)` times. The simulation
   is deterministic and works like this: time `t` starts at 0. Repeat: (a) every running task whose end time is at most
   `t` finishes and frees its worker; (b) while a worker is free and some task that has not started has all its
   dependencies finished, start the ready task with the smallest name at time `t` (it ends at `t + duration`);
   (c) if all tasks have finished stop, otherwise set `t` to the smallest end time among running tasks and repeat.
   A task with duration 0 therefore starts and ends at the same time.
