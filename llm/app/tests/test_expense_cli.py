"""Hidden acceptance tests for the expense tracker CLI (spec: llm/app/specs/expense_cli.md)."""
import datetime
import os
import subprocess
import sys

import pytest

WORKDIR = os.environ.get("APP_DIR", "/workspace")


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "e.db")


def run(db, *args, code=None):
    env = dict(os.environ, EXPENSES_DB=db)
    p = subprocess.run([sys.executable, "expenses.py", *args], cwd=WORKDIR, env=env, capture_output=True,
                       text=True, timeout=60)
    if code is not None:
        assert p.returncode == code, f"exit {p.returncode}, stdout={p.stdout!r}, stderr={p.stderr!r}"
    return p


def lines(p):
    return [x for x in p.stdout.split("\n") if x != ""]


def test_add_prints_exact_line_and_lowercases_category(db):
    p = run(db, "add", "12.5", "Food", "--date", "2024-03-05", code=0)
    assert lines(p) == ["Added #1: 12.50 food on 2024-03-05"]
    p = run(db, "add", "3", " Coffee ", "--date", "2024-03-06", code=0)
    assert lines(p) == ["Added #2: 3.00 coffee on 2024-03-06"]


def test_add_defaults_to_today(db):
    p = run(db, "add", "1", "misc", code=0)
    today = datetime.date.today().isoformat()
    assert lines(p)[0] in (f"Added #1: 1.00 misc on {today}",
                           f"Added #1: 1.00 misc on {(datetime.date.today() - datetime.timedelta(days=1)).isoformat()}",
                           f"Added #1: 1.00 misc on {(datetime.date.today() + datetime.timedelta(days=1)).isoformat()}")


@pytest.mark.parametrize("args", [
    ("abc", "food"), ("0", "food"), ("-5", "food"), ("1.234", "food"), ("5", "   "),
    ("5", "food", "--date", "2024-13-01"), ("5", "food", "--date", "2024-02-30"), ("5", "food", "--date", "yesterday"),
])
def test_invalid_add_exits_2_and_stores_nothing(db, args):
    p = run(db, "add", *args, code=2)
    assert p.stderr.strip() != ""
    assert lines(run(db, "list", code=0)) == []


def test_list_line_format_with_note(db):
    run(db, "add", "3", "Coffee", "--date", "2024-01-02", "--note", "latte, oat", code=0)
    assert lines(run(db, "list", code=0)) == ["1\t2024-01-02\t3.00\tcoffee\tlatte, oat"]


def test_list_without_note_has_empty_last_field(db):
    run(db, "add", "3", "coffee", "--date", "2024-01-02", code=0)
    assert run(db, "list", code=0).stdout == "1\t2024-01-02\t3.00\tcoffee\t\n"


def test_list_orders_by_date_then_id(db):
    run(db, "add", "1", "a", "--date", "2024-03-02", code=0)
    run(db, "add", "2", "b", "--date", "2024-03-01", code=0)
    run(db, "add", "3", "c", "--date", "2024-03-02", code=0)
    ids = [x.split("\t")[0] for x in lines(run(db, "list", code=0))]
    assert ids == ["2", "1", "3"]


def test_list_filters(db):
    run(db, "add", "1", "food", "--date", "2024-03-02", code=0)
    run(db, "add", "2", "Travel", "--date", "2024-03-09", code=0)
    run(db, "add", "3", "food", "--date", "2024-04-01", code=0)
    only_food = [x.split("\t")[0] for x in lines(run(db, "list", "--category", "FOOD", code=0))]
    assert only_food == ["1", "3"]
    march = [x.split("\t")[0] for x in lines(run(db, "list", "--month", "2024-03", code=0))]
    assert march == ["1", "2"]
    both = [x.split("\t")[0] for x in lines(run(db, "list", "--month", "2024-03", "--category", "food", code=0))]
    assert both == ["1"]
    assert lines(run(db, "list", "--month", "2023-01", code=0)) == []


def test_delete_and_unknown_id(db):
    run(db, "add", "1", "a", "--date", "2024-03-02", code=0)
    assert lines(run(db, "delete", "1", code=0)) == ["Deleted #1"]
    assert lines(run(db, "list", code=0)) == []
    p = run(db, "delete", "1", code=1)
    assert p.stderr.strip() != ""


def test_ids_are_not_reused_after_delete(db):
    run(db, "add", "1", "a", "--date", "2024-03-02", code=0)
    run(db, "delete", "1", code=0)
    assert lines(run(db, "add", "1", "a", "--date", "2024-03-02", code=0)) == ["Added #2: 1.00 a on 2024-03-02"]


CSV = (
    "date,amount,category,note\n"
    "2024-02-01,10.00,Food,lunch\n"
    '2024-02-02,5.5,transport,"bus, ""fast"" tram"\n'
    "2024-02-30,4,food,bad date\n"
    "2024-02-04,abc,food,bad amount\n"
    "2024-02-05,0,food,zero\n"
    "2024-02-06,3,,no category\n"
    "2024-02-01,10,FOOD,lunch\n"
    "2024-02-03,7.25,food,\n"
)


def test_import_counts_quotes_duplicates(db, tmp_path):
    f = tmp_path / "in.csv"
    f.write_text(CSV)
    assert lines(run(db, "import", str(f), code=0)) == ["Imported 3 rows, skipped 5"]
    rows = lines(run(db, "list", code=0))
    assert len(rows) == 3
    assert any(r.endswith('bus, "fast" tram') for r in rows)
    assert lines(run(db, "import", str(f), code=0)) == ["Imported 0 rows, skipped 8"]


def test_report_sorting_and_total(db):
    for amt, cat, d in [("10", "food", "2024-03-01"), ("2.5", "travel", "2024-03-02"),
                        ("10", "alpha", "2024-03-03"), ("7.25", "food", "2024-03-04"),
                        ("99", "food", "2024-04-01")]:
        run(db, "add", amt, cat, "--date", d, code=0)
    assert lines(run(db, "report", "--month", "2024-03", code=0)) == [
        "food: 17.25", "alpha: 10.00", "travel: 2.50", "TOTAL: 29.75"]


def test_report_uses_exact_decimals(db):
    run(db, "add", "0.1", "x", "--date", "2024-05-01", code=0)
    run(db, "add", "0.2", "x", "--date", "2024-05-02", code=0)
    assert lines(run(db, "report", "--month", "2024-05", code=0)) == ["x: 0.30", "TOTAL: 0.30"]


def test_report_for_empty_month(db):
    assert lines(run(db, "report", "--month", "2030-01", code=0)) == ["TOTAL: 0.00"]


def test_report_requires_month(db):
    run(db, "report", code=2)


@pytest.mark.parametrize("args", [(), ("frobnicate",)])
def test_no_or_unknown_command_exits_2(db, args):
    run(db, *args, code=2)


def test_amount_formatting_in_list(db):
    run(db, "add", "5", "a", "--date", "2024-03-02", code=0)
    run(db, "add", "5.5", "a", "--date", "2024-03-03", code=0)
    amounts = [x.split("\t")[2] for x in lines(run(db, "list", code=0))]
    assert amounts == ["5.00", "5.50"]
