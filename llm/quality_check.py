#!/usr/bin/env python3
"""Small executable coding check: does the served model still solve problems?

Each problem gives a function signature and hidden unit tests. The model writes the
function, the code is extracted and run against the tests in a subprocess, and the
pass rate is reported. Run it against two servers (for example MTP on and MTP off)
with the same seed and compare. Standard library only.

  python llm/quality_check.py --port 8000 --samples 4 --out llm/results_mtp.json
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROBLEMS = [
    ("two_sum", "def two_sum(nums, target):  # return indices (i, j), i < j, of the two numbers that add to target",
     "assert two_sum([2,7,11,15],9)==(0,1)\nassert two_sum([3,2,4],6)==(1,2)\nassert two_sum([3,3],6)==(0,1)"),
    ("is_balanced", "def is_balanced(s):  # True if brackets ()[]{} in s are balanced",
     "assert is_balanced('([]{})')\nassert not is_balanced('(]')\nassert is_balanced('')\nassert not is_balanced('((')"),
    ("merge_intervals", "def merge_intervals(iv):  # merge overlapping [start, end] intervals, return sorted list of lists",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]"),
    ("roman_to_int", "def roman_to_int(s):  # convert a Roman numeral to an integer",
     "assert roman_to_int('III')==3\nassert roman_to_int('LVIII')==58\nassert roman_to_int('MCMXCIV')==1994"),
    ("lis_length", "def lis_length(a):  # length of the longest strictly increasing subsequence in O(n log n)",
     "assert lis_length([10,9,2,5,3,7,101,18])==4\nassert lis_length([])==0\nassert lis_length([7,7,7])==1"),
    ("word_freq", "def word_freq(text):  # lowercase word counts as a dict, words are runs of letters/apostrophes",
     "assert word_freq(\"It's it's IT'S, a b\")=={\"it's\":3,'a':1,'b':1}\nassert word_freq('')=={}"),
    ("flatten", "def flatten(x):  # flatten arbitrarily nested lists/tuples into a flat list, keep strings whole",
     "assert flatten([1,[2,(3,[4])],'ab'])==[1,2,3,4,'ab']\nassert flatten([])==[]"),
    ("topo_sort", "def topo_sort(n, edges):  # nodes 0..n-1, edges (a,b) means a before b; return a valid order or None if cyclic",
     "o=topo_sort(4,[(0,1),(1,2),(0,3)])\nassert sorted(o)==[0,1,2,3] and o.index(0)<o.index(1)<o.index(2) and o.index(0)<o.index(3)\nassert topo_sort(2,[(0,1),(1,0)]) is None"),
    ("min_window", "def min_window(s, t):  # smallest substring of s containing all chars of t (with multiplicity), '' if none",
     "assert min_window('ADOBECODEBANC','ABC')=='BANC'\nassert min_window('a','aa')==''\nassert min_window('a','a')=='a'"),
    ("matrix_spiral", "def matrix_spiral(m):  # return the elements of a 2D list in clockwise spiral order",
     "assert matrix_spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]\nassert matrix_spiral([[1,2],[3,4],[5,6]])==[1,2,4,6,5,3]\nassert matrix_spiral([])==[]"),
]


def ask(base, model, sig, seed, max_tokens):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Complete this Python function. Reply with one ```python code block containing the full implementation and nothing else.\n\n{sig}"}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"] or ""


def passes(code_text, tests):
    m = re.search(r"```(?:python)?\n(.*?)```", code_text, re.S)
    code = m.group(1) if m else code_text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\n")
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    jobs = [(name, sig, tests, s) for name, sig, tests in PROBLEMS for s in range(args.samples)]
    with ThreadPoolExecutor(4) as ex:
        texts = list(ex.map(lambda j: ask(base, args.model, j[1], j[3], args.max_tokens), jobs))
    res = {}
    for (name, _, tests, s), t in zip(jobs, texts):
        res.setdefault(name, []).append(passes(t, tests))
    total = sum(sum(v) for v in res.values())
    n = sum(len(v) for v in res.values())
    for k, v in res.items():
        print(f"  {k:16s} {sum(v)}/{len(v)}")
    print(f"pass rate {total}/{n} = {100 * total / n:.1f}%")
    if args.out:
        json.dump({"pass": total, "n": n, "per_problem": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
