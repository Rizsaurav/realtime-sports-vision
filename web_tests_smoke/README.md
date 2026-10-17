# Page smoke tests
Regenerate the recorded API responses in `fixtures/` with: `python tests_integration/make_fixtures.py`
Run the tests from the repository root (jsdom is a dev dependency): `node --test web_tests_smoke/pages.test.mjs`
The tests load `src/sideline/web/library.html` and `match.html` in jsdom and answer the pages' `fetch` from `fixtures/`, asserting on what the user sees.
They also verify the pages make no script errors and request no URL that the fixtures do not answer.
