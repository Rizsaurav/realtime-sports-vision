# Feature: annotations — editable metadata for a stored match

## Purpose

Store user-edited metadata next to a recording: a display name for the match,
names for the two teams, names for players keyed by track id (with the ability
to merge several track ids into one named player via an alias), free notes,
tags and a favourite flag. The feature is a read/write plug-in for the library
app: it uses the feature registry (`tasks/feature_registry.md`), persists one
small JSON document per match, and exposes a pure function
`resolve_player_id` that other features import to map a track id to the
canonical track id of its named player. The feature never reads or modifies
`frames.jsonl` or `meta.json`: annotations are independent of the recorded
content (even a one-frame recording gets full CRUD).

## Interface

- Module: `src/sideline/features/annotations.py` (importable as
  `sideline.features.annotations`), a feature of the registry: it defines
  `register(router, ctx)` which adds routes to the `APIRouter` mounted under
  `/api/matches/{match_id}`. The match directory is
  `Path(ctx.store.root) / match_id`; the document is stored there as
  `annotations.json`. Standard library plus FastAPI only, no network.
- `default_annotations() -> dict`: a fresh (independent) copy of the default
  document (R1).
- `resolve_player_id(track_id, players=None) -> int`: the canonical track id
  for `track_id` (R8). `track_id` is an `int` (a bool is rejected); `players`
  is the `players` mapping of an annotations document (keys `str(track_id)`)
  or `None`. Raises `ValueError` for a bad `track_id` or an alias cycle.
  Never mutates `players`.
- The annotations document, exact top-level keys (always all six, in this
  shape when served):

  ```json
  {"match_name": null, "teams": {"left": null, "right": null},
   "players": {"7": {"name": "Kun", "alias_of": null},
               "9": {"name": null, "alias_of": null}},
   "notes": "", "tags": [], "favourite": false}
  ```

- Routes (all answers carry exactly `{"match_id": <id str>, "annotations": <doc>}`):
  - `GET /api/matches/{match_id}/annotations` → `200` with the stored
    document, or the default document when `annotations.json` is missing or
    invalid (R9).
  - `PUT /api/matches/{match_id}/annotations` → create or replace the whole
    document. The request body is a JSON object; fields not present reset to
    their defaults. `200` with the new document. Idempotent: the same PUT
    twice answers identically. `400` with `{"error": <msg>}` when invalid —
    the previous document stays unchanged (R11). `404` `{"error": <str>}`
    for an unknown or malformed match id (via `ctx.require`), checked before
    the body is parsed.
  - `PATCH /api/matches/{match_id}/annotations` → partial update: each field
    present in the body replaces the corresponding field of the current
    document (shallow merge, R6); fields absent keep their current value.
    An empty body `{}` is a no-op answering with the current document.
    `200` with the merged document; `400`/`404` as for PUT.
  - `DELETE /api/matches/{match_id}/annotations` → removes
    `annotations.json` (if present) and answers `200` with the default
    document. Idempotent: deleting twice answers the same. `404` as for PUT.
- No other status codes, no query parameters, no other files written. The
  handler reads the raw request body and parses it as JSON itself (no
  framework validation layer), so an invalid body is a `400` with
  `{"error": "body must be a JSON object"}` — never a framework 422 or 500.
- Persistence (R11): `annotations.json` is written atomically (temporary
  file in the match directory, then rename) under a process-wide lock, as
  UTF-8 JSON. The file exists if and only if the stored document differs
  from the defaults. No temporary files may be left behind.

## Rules

- R1 Defaults. The default document is exactly
  `{"match_name": null, "teams": {"left": null, "right": null},
  "players": {}, "notes": "", "tags": [], "favourite": false}`.
  `default_annotations()` returns a fresh deep copy each call.
- R2 Field limits (named constants). `MATCH_NAME_MAX = 100`,
  `TEAM_NAME_MAX = 50`, `PLAYER_NAME_MAX = 50`, `NOTES_MAX = 5000`,
  `TAG_LEN_MAX = 30`, `TAGS_MAX = 20`. All text fields are Unicode strings;
  length is measured in code points; a *control character* is any character
  with ordinal `< 32` or `== 127` and is forbidden in every text field.
  Example: `"Café"` is 4 characters and valid; `"a\nb"` is invalid because
  `ord("\n") = 10 < 32`; `"a" * 100` is a valid match name, `"a" * 101` is
  not.
- R3 match_name. `null` (no display name) or a string of 1 to 100 characters
  without control characters. Empty string is not allowed (use `null`).
  Example: `{"match_name": ""}` → 400 `match_name must be 1-100 characters`.
- R4 teams. An object with at most the keys `"left"` and `"right"`, each
  `null` or a string of 1 to 50 characters without control characters.
  Example: `{"teams": {"left": "Alpha", "centre": "Mid"}}` → 400
  `unknown team: centre`; `{"teams": {"right": "b" * 51}}` → 400
  `team name must be 1-50 characters`.
- R5 players. An object mapping track id to an entry. Keys are canonical
  decimal strings (`0` or a digit string without leading zeros), i.e.
  `str(track_id)` for an integer track id `>= 0`; anything else is invalid.
  Example: `"07"` → 400 `invalid track id key: 07`, `"0"` is valid. Each
  entry is an object with at most the keys `name` (null or a string of 1 to
  50 characters, no control characters) and `alias_of` (null or an integer
  `>= 0`; a bool is not an integer). `alias_of` is the track id this id is
  merged into. Example: `{"7": {"name": "Kun", "alias_of": 9}, "9": {}}`
  stores as `{"7": {"name": "Kun", "alias_of": 9}, "9": {"name": null,
  "alias_of": null}}`.
- R6 Merge (PATCH). The current document (defaults when the file is
  missing or invalid) and the body are shallow-merged: each top-level field
  present in the body replaces the field of the current document as a whole.
  So `PATCH {"players": {"7": {"name": "Kun"}}}` on a document with players
  `{"1": {...}, "7": {...}}` leaves exactly player `7`; `PATCH {"tags":
  ["cup"]}` keeps match_name, teams, players, notes, favourite. The merged
  document is then validated whole, including the alias checks of R7.
- R7 Alias integrity. In the document that would be stored (the PUT body
  itself, or the PATCH merge): an entry's `alias_of` must not equal the
  entry's own track id (400 `player <id>: alias_of may not be itself`), and
  the alias links must contain no cycle (400 `alias cycle detected`).
  Example: `{"players": {"1": {"alias_of": 2}, "2": {"alias_of": 1}}` →
  400 `alias cycle detected`, and the previous document is untouched. An
  alias may point to a track id with no entry (the chain simply ends there);
  it may point to a track id absent from the recording.
- R8 resolve_player_id. Let `cur = track_id`. While the entry
  `players[str(cur)]` exists and its `alias_of` is an integer `>= 0`, set
  `cur` to that `alias_of`. Return `cur`. The function tolerates
  hand-built mappings: an entry that is not an object, or whose `alias_of`
  is not an integer `>= 0` (a string, a negative number, `None`, a bool),
  ends the chain. Revisiting an id already visited raises
  `ValueError("alias cycle")`; a `track_id` that is not an integer `>= 0`
  (a bool, a negative number, a string) raises `ValueError`. Example:
  `players = {"3": {"alias_of": 4}, "4": {"alias_of": 5}, "5": {"name":
  "Five"}}` → `resolve_player_id(3) = 5`, `resolve_player_id(4) = 5`,
  `resolve_player_id(5) = 5`, `resolve_player_id(6) = 6`,
  `resolve_player_id(7, {}) = 7`, `resolve_player_id(8, None) = 8`.
- R9 Reading. A missing `annotations.json` answers with the defaults. A file
  whose content is not valid JSON, not a JSON object, or fails any
  validation rule (R3-R7 or R10, or carries unknown keys at any level) is
  treated as missing: the answer is the defaults and the file is left
  untouched on disk. A valid file is served as stored, with any missing key
  — top-level, team side, or player entry field — filled by the defaults
  (an entry `{"name": "Kun"}` is served as `{"name": "Kun", "alias_of":
  null}`; a document without a `tags` key is served with `"tags": []`).
- R10 Notes, tags, favourite. `notes`: a string of 0 to 5000 characters, no
  control characters (`""` is the default and valid). `tags`: a list of at
  most 20 strings, each 1 to 30 characters, no control characters, no
  duplicates; order is preserved as given. `favourite`: a JSON boolean.
  Example: 20 tags → 200, 21 tags → 400 `at most 20 tags`;
  `["wet", "cup", "wet"]` → 400 `duplicate tag: wet`; `"t" * 31` → 400
  `tag must be 1-30 characters`; `{"favourite": 1}` → 400
  `favourite must be a boolean` (1 is not a boolean).
- R11 Validation and persistence. The checks run in this order and at most
  one error is reported: body is a JSON object, unknown top-level field
  (alphabetically first, `unknown field: <name>`), match_name, teams
  (left, then right, then unknown team alphabetically first), players
  (invalid keys first — lexicographically — then entries in ascending track
  id order), notes, tags (length first, then elements in list order),
  favourite, and finally the R7 alias checks. Messages: `body must be a
  JSON object`, `unknown field: <name>`, `match_name must be a string or
  null`, `match_name must be 1-100 characters`, `match_name may not contain
  control characters`, `teams must be an object`, `unknown team: <name>`,
  `team name must be a string or null`, `team name must be 1-50 characters`,
  `team names may not contain control characters`, `players must be an
  object`, `invalid track id key: <key>`, `player <id>: entry must be an
  object`, `player <id>: unknown field: <name>`, `player <id>: name must be
  a string or null`, `player <id>: name must be 1-50 characters`,
  `player <id>: name may not contain control characters`, `player <id>:
  alias_of must be a non-negative integer`, `player <id>: alias_of may not
  be itself`, `alias cycle detected`, `notes must be a string`, `notes must
  be at most 5000 characters`, `notes may not contain control characters`,
  `tags must be a list`, `at most 20 tags`, `tag must be a string`, `tag
  must be 1-30 characters`, `tags may not contain control characters`,
  `duplicate tag: <tag>`, `favourite must be a boolean`. On any `400` the
  stored document (and the file, byte for byte) is unchanged. Successful
  writes are atomic (temp file + rename, process-wide lock) and leave no
  temporary files in the match directory.

## Edge cases

- Empty input: a zero-frame recording cannot be stored at all (the upload
  answers 400), so the routes never see an empty match. An empty PUT body
  `{}` is valid and resets the document to the defaults, removing
  `annotations.json` from disk.
- One frame: the smallest storable recording. Example: a 1-frame upload gets
  full CRUD — `PUT {"favourite": true}` → 200, `GET` returns
  `"favourite": true`, `DELETE` restores the defaults; none of this reads
  frames.
- Missing ball, missing positions, duplicate ids, gaps in frame numbers:
  annotations never read `frames.jsonl`, so none of this can change the
  answer. A recording whose ball is `null` in every frame, whose people
  carry `xy: null`, which lists the same track id twice in a frame, and
  whose frame numbers jump from 4 to 30 behaves exactly like a full
  recording for this API; `resolve_player_id` depends only on the stored
  players mapping.
- Huge input: a 25,000-frame recording with 2,500 distinct track ids, plus a
  PUT of a document with 2,500 player entries (25 of them carrying
  `alias_of`). Budget: the PUT completes in under 10.0 s wall clock, the
  next GET in under 2.0 s, and `resolve_player_id` follows a 10,000-link
  alias chain in under 0.5 s.
- Corrupt store: hand-written `annotations.json` that is not valid JSON (or
  a valid JSON document that fails a rule) is treated as missing (R9): GET
  answers the defaults, 200, and the file is left untouched.

## Non-goals

- No change to the matches API: `GET /api/matches/{match_id}` still returns
  the original `meta.json` name; resolving the display name (the
  annotations `match_name`, falling back to the meta name) is done by the
  page, not by this API.
- No per-player statistics, no validation that a track id, team name or tag
  refers to something in the recording, no query parameters or per-field
  routes, no editing of colours, no authentication, no caching layer (the
  file is the source of truth and is read on every GET).
