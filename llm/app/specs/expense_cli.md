# Expense tracker command line tool

I want a command line expense tracker written in Python 3, in a single file `expenses.py` in the working directory.
It is run as `python expenses.py <command> ...`. Expenses are stored in a SQLite database whose path is taken from
the environment variable `EXPENSES_DB` (default `expenses.db` in the working directory). Money must be handled with
exact decimal arithmetic. Use only the standard library.

An expense has an id (integers starting at 1, never reused), a date (`YYYY-MM-DD`), an amount (positive, at most two
decimal places), a category and an optional note. Categories are case-insensitive: trim surrounding whitespace and
store and show them in lowercase. A category must not be empty.

## Commands

- `add <amount> <category> [--date YYYY-MM-DD] [--note TEXT]` stores an expense (the date defaults to today) and
  prints `Added #<id>: <amount with 2 decimals> <category> on <date>`. An invalid amount (not a number, zero,
  negative, more than two decimals), an invalid or impossible date (such as 2024-02-30) or an empty category prints a
  message to stderr, stores nothing and exits with code 2.
- `list [--category C] [--month YYYY-MM]` prints one line per matching expense, ordered by date and then id:
  `<id>\t<date>\t<amount with 2 decimals>\t<category>\t<note>` (tab separated; the note may be empty). The category
  filter ignores case. If nothing matches it prints nothing.
- `delete <id>` prints `Deleted #<id>`. An unknown id prints a message to stderr and exits with code 1.
- `import <file.csv>` reads a CSV file with the header `date,amount,category,note` (standard CSV quoting, so a note
  may contain commas and quotes). A row is skipped if its date is invalid, its amount is not a positive number with at
  most two decimals, or its category is empty. A row is also skipped if an identical expense already exists, meaning
  the same date, the same amount value, the same category (ignoring case) and the same note. Print
  `Imported <N> rows, skipped <M>`.
- `report --month YYYY-MM` prints one line per category in that month, `<category>: <total with 2 decimals>`, ordered by
  total descending and then category name ascending, followed by the line `TOTAL: <sum with 2 decimals>`. A month
  with no expenses prints only `TOTAL: 0.00`. `--month` is required (a missing one exits with code 2).

Running with no command or an unknown command exits with code 2.
