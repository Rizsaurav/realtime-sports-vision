# Kanban board web app

I want a simple kanban board that I can open in a browser. Please build it in Python.

`python app.py` must start an HTTP server on port 8000 on all interfaces. You may use Flask, FastAPI or only the
standard library (`pip install` works). State is stored in a SQLite database file named `board.db` in the working
directory so it survives a restart. On the very first start (no database) the board has three columns, in this order:
"To Do", "In Progress" and "Done". All API responses are JSON (204 responses are empty).

## API

- `GET /api/board` returns 200 and
  `{"columns": [{"id": int, "name": str, "cards": [{"id": int, "title": str, "description": str}, ...]}, ...]}`.
  Columns are in board order and cards are in their order within the column.
- `POST /api/columns` with `{"name": str}` creates a column at the end of the board and returns 201 with
  `{"id": int, "name": str}`. An empty or missing name returns 400. A name equal to an existing one, ignoring case,
  returns 409.
- `DELETE /api/columns/<id>` deletes the column and all its cards and returns 204. An unknown id returns 404.
- `POST /api/cards` with `{"column_id": int, "title": str, "description": str}` (`description` optional, default `""`)
  appends a card to the end of that column and returns 201 with
  `{"id": int, "title": str, "description": str, "column_id": int}`. An empty or missing title returns 400. An unknown
  `column_id` returns 404.
- `PATCH /api/cards/<id>` with `{"title"?: str, "description"?: str}` updates only the given fields and returns 200
  with the card in the same format as above. An empty title returns 400. An unknown id returns 404.
- `POST /api/cards/<id>/move` with `{"column_id": int, "position": int}` moves the card into that column at that
  0-based position and returns 200 with the card. A position below 0 means 0 and a position beyond the end means the
  end. The other cards keep their relative order. An unknown card or column returns 404.
- `DELETE /api/cards/<id>` returns 204. An unknown id returns 404.

## Web page

`GET /` returns an HTML page (content type `text/html`) that shows the board in the browser, using this page markup: a
container element with `id="board"`, one visible element per column and per card, and a way to add a card to a
column. It must load the board with `fetch` from `/api/board`, and cards must be draggable between columns
(HTML5 `draggable`, with the move sent to the API). Any script or stylesheet the page loads from your server by a
path starting with `/` must be served with status 200.
