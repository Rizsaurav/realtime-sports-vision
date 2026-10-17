"""Acceptance tests for the annotations feature, built from
features/annotations/spec.md.

Every expected value is derived by hand from the spec's rules (one document
with exactly six fields, match name 1-100 chars, team/player names 1-50,
notes 0-5000, tags 1-30 of at most 20, canonical decimal track id keys,
alias_of chains without cycles, atomic persistence of annotations.json,
resolve_player_id following the alias_of chain).
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SIDELINE_SRC", "src"))

import pytest
from fastapi.testclient import TestClient

from sideline.library import create_library_app

DEFAULT_DOC = {"match_name": None, "teams": {"left": None, "right": None},
               "players": {}, "notes": "", "tags": [], "favourite": False}


# ---------- frame builders ----------

def person(pid, team, x, y, role="player"):
    return {"id": pid, "role": role, "team": team, "jersey": str(pid),
            "box": [0, 0, 10, 10], "xy": [x, y] if x is not None else None}


def ball(x, y):
    return {"box": [0, 0, 10, 10], "xy": [x, y] if x is not None else None}


def frames(n, people_fn, ball_fn, start=0):
    """n consecutive frame numbers starting at start."""
    out = []
    for f in range(start, start + n):
        out.append({"frame": f, "delay": 0,
                    "people": people_fn(f) or [],
                    "ball": ball_fn(f), "colors": None})
    return out


def small_rec():
    """30 frames, three left and three right players, static ball."""
    return frames(30,
                  lambda f: [person(1, "left", -10, 0), person(2, "left", -8, 4),
                             person(3, "left", -12, -4),
                             person(4, "right", 10, 0), person(5, "right", 8, 4),
                             person(6, "right", 12, -4)],
                  lambda f: ball(0, 0))


def one_frame_rec():
    return frames(1, lambda f: [person(1, "left", 0, 0)],
                  lambda f: ball(1, 1))


def gap_rec():
    """Frames 0-4 then 30-34 (a gap), ball missing everywhere, player 1
    without a position on odd frames and duplicated in frame 3."""
    out = []
    for f in list(range(0, 5)) + list(range(30, 35)):
        p1 = person(1, "left", None if f % 2 else 0.0, 0.0)
        people = [p1, person(2, "right", 5.0, 0.0)]
        if f == 3:
            people = [p1, person(1, "left", 9.0, 9.0), person(2, "right", 5.0, 0.0)]
        out.append({"frame": f, "delay": 0, "people": people,
                    "ball": None, "colors": None})
    return out


def big_rec():
    """25,000 frames with 10 people per frame covering ids 1..2500."""
    return frames(25000,
                  lambda f: [person((f * 10 + j) % 2500 + 1,
                                    "left" if j % 2 == 0 else "right",
                                    float(j), 0.0) for j in range(10)],
                  lambda f: None)


# ---------- app / store helpers ----------

@pytest.fixture()
def client(tmp_path):
    return TestClient(create_library_app(tmp_path))


def upload(client, rec, name="rec"):
    data = ("\n".join(json.dumps(f) for f in rec) + "\n").encode("utf-8")
    r = client.post("/api/matches",
                    files={"file": ("rec.jsonl", data, "application/octet-stream")},
                    data={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def ann_url(mid):
    return "/api/matches/%s/annotations" % mid


def get_ann(client, mid):
    return client.get(ann_url(mid))


def put_ann(client, mid, doc):
    return client.put(ann_url(mid), json=doc)


def put_raw(client, mid, content):
    return client.put(ann_url(mid), content=content,
                      headers={"Content-Type": "application/json"})


def patch_ann(client, mid, doc):
    return client.patch(ann_url(mid), json=doc)


def delete_ann(client, mid):
    return client.delete(ann_url(mid))


def ann_file(tmp_path, mid):
    return tmp_path / mid / "annotations.json"


# ---------- GET: defaults ----------

def test_get_defaults_for_fresh_match(tmp_path, client):
    mid = upload(client, small_rec())
    r = get_ann(client, mid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"match_id", "annotations"}
    assert body["match_id"] == mid
    assert body["annotations"] == DEFAULT_DOC
    assert not ann_file(tmp_path, mid).exists()


def test_default_document_shape(client):
    mid = upload(client, one_frame_rec())
    doc = get_ann(client, mid).json()["annotations"]
    assert set(doc) == {"match_name", "teams", "players", "notes",
                        "tags", "favourite"}
    assert set(doc["teams"]) == {"left", "right"}
    assert doc["match_name"] is None
    assert doc["teams"]["left"] is None and doc["teams"]["right"] is None
    assert doc["players"] == {}
    assert doc["notes"] == "" and doc["tags"] == []
    assert doc["favourite"] is False  # a real boolean, not 0


# ---------- PUT ----------

def test_put_full_document_roundtrip(tmp_path, client):
    mid = upload(client, small_rec())
    doc = {"match_name": "Cup final", "teams": {"left": "Alpha", "right": None},
           "players": {"7": {"name": "Kun", "alias_of": 9}, "9": {}},
           "notes": "Rain in the second half", "tags": ["cup", "wet"],
           "favourite": True}
    r = put_ann(client, mid, doc)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"match_id", "annotations"}
    assert body["match_id"] == mid
    # the empty entry for 9 is stored with its full shape
    assert body["annotations"] == {
        "match_name": "Cup final",
        "teams": {"left": "Alpha", "right": None},
        "players": {"7": {"name": "Kun", "alias_of": 9},
                    "9": {"name": None, "alias_of": None}},
        "notes": "Rain in the second half", "tags": ["cup", "wet"],
        "favourite": True}
    assert get_ann(client, mid).json()["annotations"] == body["annotations"]
    assert json.loads(ann_file(tmp_path, mid).read_text(encoding="utf-8")) \
        == body["annotations"]


def test_put_missing_fields_reset_to_defaults(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid, {"match_name": "Only name"})
    assert r.status_code == 200, r.text
    assert r.json()["annotations"] == {**DEFAULT_DOC, "match_name": "Only name"}


def test_put_empty_body_resets_and_removes_file(tmp_path, client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"favourite": True}).status_code == 200
    assert ann_file(tmp_path, mid).is_file()
    r = put_ann(client, mid, {})
    assert r.status_code == 200, r.text
    assert r.json()["annotations"] == DEFAULT_DOC
    assert not ann_file(tmp_path, mid).exists()
    assert get_ann(client, mid).json()["annotations"] == DEFAULT_DOC


def test_put_is_idempotent(client):
    mid = upload(client, one_frame_rec())
    doc = {"match_name": "Twice", "favourite": True}
    b1 = put_ann(client, mid, doc)
    b2 = put_ann(client, mid, doc)
    assert b1.status_code == 200 and b2.status_code == 200
    assert b1.json() == b2.json()


def test_put_does_not_touch_match_files(tmp_path, client):
    mid = upload(client, small_rec())
    before_meta = (tmp_path / mid / "meta.json").read_bytes()
    before_frames = (tmp_path / mid / "frames.jsonl").read_bytes()
    assert put_ann(client, mid, {"match_name": "X", "favourite": True}).status_code == 200
    assert (tmp_path / mid / "meta.json").read_bytes() == before_meta
    assert (tmp_path / mid / "frames.jsonl").read_bytes() == before_frames


def test_failed_put_keeps_previous_document(tmp_path, client):
    mid = upload(client, one_frame_rec())
    good = {"match_name": "Keep me", "notes": "n"}
    assert put_ann(client, mid, good).status_code == 200
    stored_before = ann_file(tmp_path, mid).read_bytes()
    r = put_ann(client, mid, {"notes": "x" * 5001})
    assert r.status_code == 400
    assert r.json() == {"error": "notes must be at most 5000 characters"}
    assert ann_file(tmp_path, mid).read_bytes() == stored_before
    assert get_ann(client, mid).json()["annotations"] \
        == {**DEFAULT_DOC, "match_name": "Keep me", "notes": "n"}


def test_failed_patch_keeps_previous_document(client):
    mid = upload(client, one_frame_rec())
    good = {"match_name": "Keep me", "players": {"1": {"alias_of": 2}}}
    assert put_ann(client, mid, good).status_code == 200
    r = patch_ann(client, mid, {"notes": "x" * 5001})
    assert r.status_code == 400
    assert r.json() == {"error": "notes must be at most 5000 characters"}
    doc = get_ann(client, mid).json()["annotations"]
    assert doc == {**DEFAULT_DOC, "match_name": "Keep me",
                   "players": {"1": {"name": None, "alias_of": 2}}}


# ---------- PATCH ----------

def test_patch_shallow_merge(client):
    mid = upload(client, small_rec())
    base = {"match_name": "M", "teams": {"left": "L"},
            "players": {"1": {"name": "One"}, "7": {"name": "Seven"}},
            "notes": "n", "tags": ["a", "b"], "favourite": True}
    assert put_ann(client, mid, base).status_code == 200
    r = patch_ann(client, mid, {"tags": ["cup"]})
    assert r.status_code == 200, r.text
    stored = r.json()["annotations"]
    assert stored["tags"] == ["cup"]
    assert stored["match_name"] == "M"
    assert stored["teams"] == {"left": "L", "right": None}
    assert stored["players"] == {"1": {"name": "One", "alias_of": None},
                                 "7": {"name": "Seven", "alias_of": None}}
    assert stored["notes"] == "n" and stored["favourite"] is True
    # a players field replaces the whole mapping (shallow merge)
    r = patch_ann(client, mid, {"players": {"7": {"name": "Seven"}}})
    assert r.status_code == 200
    assert r.json()["annotations"]["players"] \
        == {"7": {"name": "Seven", "alias_of": None}}


def test_patch_empty_body_is_noop(tmp_path, client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"favourite": True}).status_code == 200
    before = ann_file(tmp_path, mid).read_bytes()
    r = patch_ann(client, mid, {})
    assert r.status_code == 200, r.text
    assert r.json()["annotations"] == {**DEFAULT_DOC, "favourite": True}
    assert ann_file(tmp_path, mid).read_bytes() == before


def test_patch_unknown_field_400_and_keeps_previous(client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"match_name": "Stays"}).status_code == 200
    r = patch_ann(client, mid, {"bogus": 1})
    assert r.status_code == 400
    assert r.json() == {"error": "unknown field: bogus"}
    assert get_ann(client, mid).json()["annotations"] \
        == {**DEFAULT_DOC, "match_name": "Stays"}


# ---------- DELETE ----------

def test_delete_returns_defaults_and_is_idempotent(tmp_path, client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"favourite": True}).status_code == 200
    assert ann_file(tmp_path, mid).is_file()
    r1 = delete_ann(client, mid)
    assert r1.status_code == 200, r1.text
    assert r1.json() == {"match_id": mid, "annotations": DEFAULT_DOC}
    assert not ann_file(tmp_path, mid).exists()
    r2 = delete_ann(client, mid)
    assert r2.status_code == 200 and r2.json() == r1.json()
    assert get_ann(client, mid).json()["annotations"] == DEFAULT_DOC


def test_match_delete_removes_annotations(tmp_path, client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"favourite": True}).status_code == 200
    assert client.delete("/api/matches/%s" % mid).status_code == 204
    assert not (tmp_path / mid).exists()
    assert get_ann(client, mid).status_code == 404


# ---------- 404s ----------

def test_unknown_match_404(client):
    mid = "m_0000000000"
    assert get_ann(client, mid).status_code == 404
    r = put_ann(client, mid, {"favourite": True})
    assert r.status_code == 404 and "error" in r.json()
    r = patch_ann(client, mid, {"favourite": True})
    assert r.status_code == 404 and "error" in r.json()
    assert delete_ann(client, mid).status_code == 404
    # the match check wins over body validation
    r = put_ann(client, mid, {"favourite": "yes"})
    assert r.status_code == 404 and "error" in r.json()


def test_malformed_match_id_404(client):
    mid = "m_123456789"  # 9 hex digits
    assert get_ann(client, mid).status_code == 404
    r = put_ann(client, mid, {"favourite": True})
    assert r.status_code == 404 and "error" in r.json()
    assert delete_ann(client, mid).status_code == 404


def test_traversal_match_id_404_status_only(client):
    # an encoded slash never matches a single-segment path parameter, so the
    # body is the framework's default 404, not the feature's: status only.
    mid = "m_..%2F..%2Fx"
    assert get_ann(client, mid).status_code == 404
    assert put_ann(client, mid, {"favourite": True}).status_code == 404


# ---------- field validation: match_name (R2, R3) ----------

def test_match_name_length_and_type(client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"match_name": "a" * 100}).status_code == 200
    r = put_ann(client, mid, {"match_name": "a" * 101})
    assert r.status_code == 400
    assert r.json() == {"error": "match_name must be 1-100 characters"}
    r = put_ann(client, mid, {"match_name": ""})
    assert r.status_code == 400
    assert r.json() == {"error": "match_name must be 1-100 characters"}
    r = put_ann(client, mid, {"match_name": 7})
    assert r.status_code == 400
    assert r.json() == {"error": "match_name must be a string or null"}
    assert put_ann(client, mid, {"match_name": "Café ✓"}).status_code == 200
    # the rejected writes left the last successful one in place
    assert get_ann(client, mid).json()["annotations"]["match_name"] == "Café ✓"


def test_match_name_control_characters(client):
    mid = upload(client, one_frame_rec())
    for bad in ("a\nb", "a\tb", "a\x7fb"):
        r = put_ann(client, mid, {"match_name": bad})
        assert r.status_code == 400
        assert r.json() == {"error": "match_name may not contain control characters"}
    assert put_ann(client, mid, {"match_name": "a b"}).status_code == 200


# ---------- field validation: teams (R4) ----------

def test_team_name_validation(client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"teams": {"left": "a" * 50}}).status_code == 200
    r = put_ann(client, mid, {"teams": {"left": "a" * 51}})
    assert r.status_code == 400
    assert r.json() == {"error": "team name must be 1-50 characters"}
    r = put_ann(client, mid, {"teams": {"centre": "Mid"}})
    assert r.status_code == 400
    assert r.json() == {"error": "unknown team: centre"}
    r = put_ann(client, mid, {"teams": {"left": 3}})
    assert r.status_code == 400
    assert r.json() == {"error": "team name must be a string or null"}
    r = put_ann(client, mid, {"teams": "left"})
    assert r.status_code == 400
    assert r.json() == {"error": "teams must be an object"}
    r = put_ann(client, mid, {"teams": {"left": "A\nB"}})
    assert r.status_code == 400
    assert r.json() == {"error": "team names may not contain control characters"}
    r = put_ann(client, mid, {"teams": {"left": "Alpha", "right": None}})
    assert r.status_code == 200
    assert r.json()["annotations"]["teams"] == {"left": "Alpha", "right": None}


# ---------- field validation: players (R5, R7) ----------

def test_players_key_validation(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid, {"players": {"0": {"name": "Zero"}}})
    assert r.status_code == 200
    assert r.json()["annotations"]["players"] \
        == {"0": {"name": "Zero", "alias_of": None}}
    for bad in ("07", "abc", "-3"):
        r = put_ann(client, mid, {"players": {bad: {}}})
        assert r.status_code == 400
        assert r.json() == {"error": "invalid track id key: %s" % bad}
    r = put_ann(client, mid, {"players": 7})
    assert r.status_code == 400
    assert r.json() == {"error": "players must be an object"}


def test_players_entry_validation(client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid,
                   {"players": {"7": {"name": "a" * 50}}}).status_code == 200
    r = put_ann(client, mid, {"players": {"7": {"name": "a" * 51}}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: name must be 1-50 characters"}
    r = put_ann(client, mid, {"players": {"7": "Kun"}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: entry must be an object"}
    r = put_ann(client, mid, {"players": {"7": {"role": "x"}}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: unknown field: role"}
    r = put_ann(client, mid, {"players": {"7": {"name": 7}}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: name must be a string or null"}
    r = put_ann(client, mid, {"players": {"7": {"name": "A\nB"}}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: name may not contain control characters"}
    for bad_alias in ("9", -1, True):
        r = put_ann(client, mid, {"players": {"7": {"alias_of": bad_alias}}})
        assert r.status_code == 400
        assert r.json() == {"error": "player 7: alias_of must be a non-negative integer"}


def test_player_self_alias_400(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid, {"players": {"7": {"alias_of": 7}}})
    assert r.status_code == 400
    assert r.json() == {"error": "player 7: alias_of may not be itself"}


def test_player_alias_cycle_400(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid,
                {"players": {"1": {"alias_of": 2}, "2": {"alias_of": 1}}})
    assert r.status_code == 400
    assert r.json() == {"error": "alias cycle detected"}
    r = put_ann(client, mid,
                {"players": {"1": {"alias_of": 2}, "2": {"alias_of": 3},
                             "3": {"alias_of": 1}}})
    assert r.status_code == 400
    assert r.json() == {"error": "alias cycle detected"}
    # a plain chain is fine
    r = put_ann(client, mid,
                {"players": {"1": {"alias_of": 2}, "2": {"alias_of": 3}}})
    assert r.status_code == 200


def test_valid_alias_chain_stored_and_resolved(client):
    from sideline.features.annotations import resolve_player_id
    mid = upload(client, small_rec())
    doc = {"players": {"3": {"alias_of": 4}, "4": {"alias_of": 5},
                       "5": {"name": "Five"}}}
    r = put_ann(client, mid, doc)
    assert r.status_code == 200, r.text
    players = r.json()["annotations"]["players"]
    assert players == {"3": {"name": None, "alias_of": 4},
                       "4": {"name": None, "alias_of": 5},
                       "5": {"name": "Five", "alias_of": None}}
    assert resolve_player_id(3, players) == 5
    assert resolve_player_id(4, players) == 5
    assert resolve_player_id(5, players) == 5
    assert resolve_player_id(6, players) == 6


# ---------- field validation: notes, tags, favourite (R10) ----------

def test_notes_validation(client):
    mid = upload(client, one_frame_rec())
    assert put_ann(client, mid, {"notes": "x" * 5000}).status_code == 200
    r = put_ann(client, mid, {"notes": "x" * 5001})
    assert r.status_code == 400
    assert r.json() == {"error": "notes must be at most 5000 characters"}
    r = put_ann(client, mid, {"notes": 12})
    assert r.status_code == 400
    assert r.json() == {"error": "notes must be a string"}
    r = put_ann(client, mid, {"notes": "a\nb"})
    assert r.status_code == 400
    assert r.json() == {"error": "notes may not contain control characters"}
    r = put_ann(client, mid, {"notes": ""})
    assert r.status_code == 200 and r.json()["annotations"]["notes"] == ""


def test_tags_validation(client):
    mid = upload(client, one_frame_rec())
    twenty = ["t" * i for i in range(1, 21)]
    assert len(twenty) == 20
    r = put_ann(client, mid, {"tags": twenty})
    assert r.status_code == 200 and r.json()["annotations"]["tags"] == twenty
    r = put_ann(client, mid, {"tags": twenty + ["u"]})
    assert r.status_code == 400
    assert r.json() == {"error": "at most 20 tags"}
    for tags, msg in (([""], "tag must be 1-30 characters"),
                      (["t" * 31], "tag must be 1-30 characters"),
                      ([7], "tag must be a string"),
                      (["a\nb"], "tags may not contain control characters"),
                      (["wet", "cup", "wet"], "duplicate tag: wet")):
        r = put_ann(client, mid, {"tags": tags})
        assert r.status_code == 400
        assert r.json() == {"error": msg}, r.text
    r = put_ann(client, mid, {"tags": "wet"})
    assert r.status_code == 400
    assert r.json() == {"error": "tags must be a list"}
    r = put_ann(client, mid, {"tags": ["wet", "cup"]})
    assert r.status_code == 200
    assert r.json()["annotations"]["tags"] == ["wet", "cup"]


def test_favourite_must_be_boolean(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid, {"favourite": True})
    assert r.status_code == 200 and r.json()["annotations"]["favourite"] is True
    for bad in (1, 0, "yes", "true"):
        r = put_ann(client, mid, {"favourite": bad})
        assert r.status_code == 400
        assert r.json() == {"error": "favourite must be a boolean"}
    assert get_ann(client, mid).json()["annotations"]["favourite"] is True


# ---------- body handling (R11) ----------

def test_body_must_be_json_object(client):
    mid = upload(client, one_frame_rec())
    for content in (b"[1, 2]", b"42", b"not json {", b"null"):
        r = put_raw(client, mid, content)
        assert r.status_code == 400, r.text
        assert r.json() == {"error": "body must be a JSON object"}
    r = client.put(ann_url(mid))  # no body at all
    assert r.status_code == 400
    assert r.json() == {"error": "body must be a JSON object"}
    # an empty object IS a valid JSON object (the reset)
    assert put_raw(client, mid, b"{}").status_code == 200
    assert client.patch(ann_url(mid),
                        content=b"{}",
                        headers={"Content-Type": "application/json"}
                        ).status_code == 200


# ---------- on-disk document (R9) ----------

def test_corrupt_file_treated_as_defaults(tmp_path, client):
    mid = upload(client, one_frame_rec())
    f = ann_file(tmp_path, mid)
    f.write_bytes(b"{ not json")
    r = get_ann(client, mid)
    assert r.status_code == 200, r.text
    assert r.json()["annotations"] == DEFAULT_DOC
    assert f.read_bytes() == b"{ not json"  # left untouched
    # valid JSON that fails a rule (21 tags) counts as corrupt too
    f.write_text(json.dumps({**DEFAULT_DOC, "tags": ["t"] * 21}),
                 encoding="utf-8")
    r = get_ann(client, mid)
    assert r.status_code == 200
    assert r.json()["annotations"] == DEFAULT_DOC


def test_handwritten_valid_file_served_normalized(tmp_path, client):
    mid = upload(client, one_frame_rec())
    f = ann_file(tmp_path, mid)
    f.write_text(json.dumps({"match_name": "Hand",
                             "players": {"7": {"name": "Kun"}}}),
                 encoding="utf-8")
    doc = get_ann(client, mid).json()["annotations"]
    assert doc == {**DEFAULT_DOC, "match_name": "Hand",
                   "players": {"7": {"name": "Kun", "alias_of": None}}}


def test_no_leftover_temp_files(tmp_path, client):
    mid = upload(client, one_frame_rec())
    for doc in ({"favourite": True}, {"match_name": "n"}, {}):
        assert put_ann(client, mid, doc).status_code == 200
    names = {str(p.relative_to(tmp_path / mid))
             for p in (tmp_path / mid).rglob("*") if p.is_file()}
    assert names == {"frames.jsonl", "meta.json"}


# ---------- edge cases: recording content is irrelevant ----------

def test_one_frame_recording_crud(client):
    mid = upload(client, one_frame_rec())
    r = put_ann(client, mid, {"favourite": True, "match_name": "tiny"})
    assert r.status_code == 200, r.text
    assert get_ann(client, mid).json()["annotations"] \
        == {**DEFAULT_DOC, "favourite": True, "match_name": "tiny"}
    r = patch_ann(client, mid, {"tags": ["tiny"]})
    assert r.status_code == 200
    doc = r.json()["annotations"]
    assert doc["tags"] == ["tiny"] and doc["match_name"] == "tiny"
    assert delete_ann(client, mid).status_code == 200
    assert get_ann(client, mid).json()["annotations"] == DEFAULT_DOC


def test_annotations_ignore_frames_content(client):
    # gap in frame numbers, no ball anywhere, a person without a position,
    # a duplicated track id in one frame: none of this is read, so the API
    # behaves exactly as for a full recording.
    mid = upload(client, gap_rec())
    rec = gap_rec()
    assert rec[4]["frame"] == 4 and rec[5]["frame"] == 30  # the gap is real
    assert [f for f in rec if f["ball"] is not None] == []
    r = put_ann(client, mid, {"players": {"1": {"name": "One"}}})
    assert r.status_code == 200, r.text
    from sideline.features.annotations import resolve_player_id
    assert resolve_player_id(1, r.json()["annotations"]["players"]) == 1
    assert get_ann(client, mid).json()["annotations"]["players"] \
        == {"1": {"name": "One", "alias_of": None}}


# ---------- resolve_player_id (R8) ----------

def test_resolve_player_id():
    from sideline.features.annotations import resolve_player_id
    players = {"3": {"alias_of": 4}, "4": {"alias_of": 5},
               "5": {"name": "Five"}}
    assert resolve_player_id(3, players) == 5
    assert resolve_player_id(4, players) == 5
    assert resolve_player_id(5, players) == 5
    assert resolve_player_id(6, players) == 6
    assert resolve_player_id(7, {}) == 7
    assert resolve_player_id(8, None) == 8
    assert resolve_player_id(0, {}) == 0
    # an alias may point outside the mapping: the chain ends there
    assert resolve_player_id(7, {"7": {"alias_of": 9}}) == 9
    # malformed alias values end the chain instead of raising
    for entry in ({"alias_of": "9"}, {"alias_of": -2}, {"alias_of": None},
                  {"alias_of": True}, {"name": "x"}, "Kun", None):
        assert resolve_player_id(7, {"7": entry}) == 7
    long_chain = {str(i): {"alias_of": i + 1} for i in range(1, 10001)}
    assert resolve_player_id(1, long_chain) == 10001


def test_resolve_player_id_validation_errors():
    from sideline.features.annotations import resolve_player_id
    for bad_id in (True, -1, "7", None):
        with pytest.raises(ValueError):
            resolve_player_id(bad_id, {})
    with pytest.raises(ValueError):
        resolve_player_id(1, {"1": {"alias_of": 2}, "2": {"alias_of": 1}})
    with pytest.raises(ValueError):
        resolve_player_id(1, {"1": {"alias_of": 1}})
    with pytest.raises(ValueError):
        resolve_player_id(1, {"1": {"alias_of": 2}, "2": {"alias_of": 3},
                              "3": {"alias_of": 1}})


def test_resolve_player_id_no_mutation():
    from sideline.features.annotations import resolve_player_id
    players = {"3": {"alias_of": 4}, "4": {"alias_of": 5}}
    snapshot = copy.deepcopy(players)
    resolve_player_id(3, players)
    assert players == snapshot


def test_default_annotations_fresh_copies():
    from sideline.features.annotations import default_annotations
    d1 = default_annotations()
    d2 = default_annotations()
    assert d1 == DEFAULT_DOC and d2 == DEFAULT_DOC
    assert d1 is not d2
    assert d1["teams"] is not d2["teams"]
    assert d1["players"] is not d2["players"]
    assert d1["tags"] is not d2["tags"]
    d1["tags"].append("x")
    d1["teams"]["left"] = "L"
    assert default_annotations() == DEFAULT_DOC


# ---------- performance (edge case: huge input) ----------

def test_performance_25000_frames(client):
    from sideline.features.annotations import resolve_player_id
    rec = big_rec()
    assert len(rec) == 25000
    ids = {p["id"] for f in rec for p in f["people"]}
    assert len(ids) == 2500 and min(ids) == 1 and max(ids) == 2500
    mid = upload(client, rec)
    players = {}
    for i in range(1, 2501):
        entry = {"name": "P%d" % i}
        if i % 100 == 0:  # 25 pairwise aliases, no cycles
            entry["alias_of"] = i - 1
        players[str(i)] = entry
    t0 = time.monotonic()
    r = put_ann(client, mid, {"players": players})
    put_s = time.monotonic() - t0
    assert r.status_code == 200, r.text
    assert put_s < 10.0
    stored = r.json()["annotations"]
    assert stored["players"]["100"] == {"name": "P100", "alias_of": 99}
    assert stored["players"]["101"] == {"name": "P101", "alias_of": None}
    t0 = time.monotonic()
    r2 = get_ann(client, mid)
    get_s = time.monotonic() - t0
    assert get_s < 2.0
    assert r2.status_code == 200
    assert r2.json() == r.json()
    chain = {str(i): {"alias_of": i + 1} for i in range(1, 10001)}
    t0 = time.monotonic()
    out = resolve_player_id(1, chain)
    assert time.monotonic() - t0 < 0.5
    assert out == 10001
