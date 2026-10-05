"""SportsMOT loader (MOT-format clips: player boxes + track IDs).

Dataset: https://github.com/MCG-NJU/SportsMOT-Dataset (240 clips, MOT format)
Layout expected under data/raw/sportsmot/<split>/<seq>/:
    seqinfo.ini, img1/<frame>.jpg, gt/gt.txt, det/det.txt (optional)

gt.txt rows: frame_id, track_id, x, y, w, h, conf, class_id, visibility
(CPU + network only: downloading, parsing.)
"""

from pathlib import Path


class SportsMOT:
    def __init__(self, root="data/raw/sportsmot", split="test"):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Run bash scripts/download_data.sh first."
            )

    def sequences(self):
        """Yield sequence names that have an img1/ frames directory."""
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "img1").exists()
        )

    def frames(self, seq_name):
        """Sorted list of frame image paths for a sequence."""
        img_dir = self.root / seq_name / "img1"
        return sorted(img_dir.glob("*.jpg"))

    def ground_truth(self, seq_name):
        """Return dict frame_id -> list of (track_id, x1, y1, x2, y2).

        Parses gt/gt.txt (MOT challenge format: x, y, w, h top-left origin).
        """
        gt_path = self.root / seq_name / "gt" / "gt.txt"
        gt = {}
        if not gt_path.exists():
            return gt
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                fid, tid = int(float(parts[0])), int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in range(2, 6))
                gt.setdefault(fid, []).append((tid, x, y, x + w, y + h))
        return gt

    def seq_info(self, seq_name):
        """Parse seqinfo.ini -> dict (frame count, fps, resolution)."""
        info = {}
        ini = self.root / seq_name / "seqinfo.ini"
        if ini.exists():
            section = False
            for line in ini.read_text().splitlines():
                line = line.strip()
                if line == "[Sequence]":
                    section = True
                    continue
                if section and "=" in line and not line.startswith("["):
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
        return info
