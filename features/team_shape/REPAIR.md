# Repair notes: team_shape

Both failing-on-every-attempt tests (`test_seconds_relative_to_first_frame`,
`test_performance_25000_frames_within_budget`) are test bugs, not spec
ambiguities: every expected value was recomputed by hand from `spec.md` and
cross-checked against a straightforward reference implementation, which passes
all 34 tests. The spec itself is consistent except for two stale numbers in
its performance budget, corrected below.

- `test_seconds_relative_to_first_frame`: the fixture `range(25, 50)` is 25 frames = exactly one second, so with `first = 25` every frame maps to `t = 0` and the series is `[0]`, never `[0, 1]`; no implementation of the spec's `t = (frame - first) // 25` could satisfy the old assertion, so the fixture is now `range(25, 75)` (50 frames = two seconds, answer `[0, 1]` for relative binning, `[1, 2]` for the absolute-indexing mistake) and the stale comment was fixed.
- `big_recording` (performance-test fixture): the left outfielder loop used `range(11)`, so each frame carried 23 people (11 + 1 left, 10 + 1 right), contradicting the test's own assertion of 22 people and the spec's "22 people per frame"; the loop is now `range(10)`, giving 22 people per frame (10 outfielders + 1 goalkeeper per team).
- `spec.md` rule 15: "62,338,977 bytes" is the size of the 23-people recording the old fixture built, not of the 22-people recording the rule names (59,693,095 bytes), and "~416,557-char document" did not match the actual response (~239,593 chars); both figures corrected to the measured 22-people values and the machine assumption (one core, Python 3.9, laptop CPU) stated — the straightforward reference implementation measures ~3 s for the measured path there, so the kept 30 s budget is ~10× the measured cost, above the 3× floor.
- `spec.md` "Huge input" edge case: same stale byte count as rule 15, 62,338,977 → 59,693,095, to match the 22-people recording the test uploads.
