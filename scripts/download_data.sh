#!/usr/bin/env bash
# Download evaluation data. CPU + network only. No GPU.
set -euo pipefail
mkdir -p data/raw

echo "[1/2] SportsMOT (MOT-format clips)"
# https://github.com/MCG-NJU/SportsMOT-Dataset — follow repo instructions;
# expected layout: data/raw/sportsmot/test/<seq>/img1/*.jpg + gt/gt.txt
echo "  -> clone/download per the SportsMOT-Dataset README into data/raw/sportsmot/"

echo "[2/2] SoccerNet-Tracking (broadcast matches)"
# https://www.soccer-net.org/ — pip install SoccerNet, use the official downloader:
#   soccernet-downloader --task tracking --split test --out data/raw/soccernet/
echo "  -> use the SoccerNet official downloader into data/raw/soccernet/"

echo "Done. Verify a clip plays: ffplay data/raw/sportsmot/test/<seq>/img1/000001.jpg"
