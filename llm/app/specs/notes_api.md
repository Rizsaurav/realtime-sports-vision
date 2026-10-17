# Notes service

I need a small notes service I can run on a server. Please build it in Python.

`python app.py` must start an HTTP server listening on port 8000 on all interfaces. You may use Flask, FastAPI or
only the standard library (`pip install` works). All notes are stored in a SQLite database file named `notes.db` in
the working directory, so they survive a restart. Every response body is JSON (except 204, which is empty).

## Endpoints

- `GET /health` returns 200 and `{"status": "ok"}`.
- `POST /notes` creates a note from a JSON body:
  `{"title": string, "body": string, "tags": [string, ...]}`.
  `title` is required and must be 1 to 100 characters. `body` is optional (default `""`) and must be a string.
  `tags` is optional (default `[]`) and must be a list of strings. Success returns 201 and the note
  `{"id": int, "title": ..., "body": ..., "tags": [...], "created_at": "<ISO 8601 timestamp>"}`.
  Any invalid input, including a body that is not valid JSON, returns 400 with `{"error": "<message>"}`.
- `GET /notes/<id>` returns 200 and the note. An unknown id returns 404 with `{"error": "<message>"}`.
- `GET /notes` returns 200 and `{"items": [note, ...], "total": int}`. Query parameters, all optional:
  - `q`: keep notes whose title or body contains this text, ignoring case. Quotes, `%` and `_` in `q` are ordinary
    characters and must match literally (so `q=100%` matches "100% done" but not "1000 done").
  - `tag`: keep notes that have exactly this tag ("py" does not match "python").
  - `q` and `tag` together mean both must match.
  - `limit` (default 10, at most 100) and `offset` (default 0). A value that is not an integer, a negative
    `offset`, a `limit` below 1 or a `limit` above 100 returns 400.
  - Results are ordered by id ascending. `total` is the number of matching notes before `limit` and `offset`.
- `PUT /notes/<id>` replaces a note with a body in the same format as `POST /notes` and returns 200 and the updated
  note (the id and `created_at` do not change). Errors are the same as above: 400 for invalid input, 404 for an
  unknown id.
- `DELETE /notes/<id>` returns 204 with an empty body. Deleting an unknown id returns 404.

Unicode text (for example "héllo 😀") must round-trip unchanged.
