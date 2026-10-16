# URL shortener service

I need a URL shortening service. Please build it in Python.

`python app.py` starts an HTTP server on the port given by the environment variable `PORT` (default 8050), on all
interfaces. Flask, FastAPI or only the standard library are fine; a virtual environment with Flask, requests and
pytest is already active and `pip install` works inside it. Links are stored in a SQLite database file named
`links.db` in the working directory, so they survive a restart. Every response is JSON except redirects.

## Endpoints

- `GET /health` returns 200 and `{"status": "ok"}`.
- `POST /shorten` with a JSON body `{"url": string, "custom_code": string, "ttl_seconds": integer}` creates a link.
  - `url` is required. It must be an absolute `http` or `https` URL that has a host, at most 2048 characters long.
  - `custom_code` is optional. It must be 3 to 32 characters from `A-Z a-z 0-9 _ -`, and must not be one of the
    reserved words `api`, `shorten` or `health`. A code that is already in use returns 409. Codes are case sensitive
    (`AbC` and `abc` are different codes).
  - Without `custom_code`, generate a random code of exactly 7 letters and digits that is not in use.
  - `ttl_seconds` is optional. It must be an integer from 1 to 31536000 (booleans and floats are not integers); the
    link expires that many seconds after it was created.
  - Success returns 201 and
    `{"code": string, "short_url": string, "url": string, "created_at": ISO-8601 string, "expires_at": ISO-8601 string or null}`.
    `short_url` is `http://<value of the Host request header>/<code>`.
  - Any invalid input returns 400 with `{"error": "<message>"}`: a missing or invalid `url`, another scheme such as
    `ftp` or `javascript`, a bad `custom_code` or `ttl_seconds`, or a body that is not valid JSON.
  - Shortening the same URL twice creates two separate links.
- `GET /<code>` redirects to the original URL with status 302 and a `Location` header, and counts one click.
  An unknown code returns 404 with `{"error": "<message>"}`. An expired link returns 410 with
  `{"error": "<message>"}` and the click is not counted. The `Location` value is exactly the URL that was submitted.
- `GET /api/links/<code>` returns 200 and
  `{"code": string, "url": string, "clicks": integer, "created_at": string, "expires_at": string or null}`.
  An unknown code returns 404. Expired links still return their statistics.
- `DELETE /api/links/<code>` returns 204 with an empty body; an unknown code returns 404. After deletion
  `GET /<code>` returns 404 and the same code may be used again as a `custom_code`.
