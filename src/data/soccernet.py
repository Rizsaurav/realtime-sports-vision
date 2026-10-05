"""SoccerNet-Tracking helpers (broadcast match footage).

Dataset: https://www.soccer-net.org/ — 500+ games; use the Tracking split via
the official SoccerNet downloader (`pip install SoccerNet`):
    soccernet-downloader --task tracking --split test --out data/raw/soccernet/

Also handy later: SoccerNet action-spotting timestamps for the event layer
(goals, cards, corners) on top of tracking.

Clip cutting uses ffmpeg if available, else falls back to cv2.
"""

import shutil
import subprocess
from pathlib import Path


class SoccerNetTracking:
    def __init__(self, root="data/raw/soccernet", split="test"):
        self.root = Path(root) / split

    def games(self):
        """Yield game directory names present locally."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def find_video(self, game_id):
        """Locate the main video file for a game (tries common names)."""
        gdir = self.root / game_id
        for name in ("video.mp4", "1_720p.mkv", "2_720p.mkv"):
            cand = gdir / name
            if cand.exists():
                return cand
        vids = sorted(gdir.glob("*.mp4")) + sorted(gdir.glob("*.mkv"))
        return vids[0] if vids else None

    def sample_clip(self, game_id, start_sec, duration_sec, out_path):
        """Cut a demo/eval clip. Prefers ffmpeg, falls back to cv2. CPU work."""
        video = self.find_video(game_id)
        if video is None:
            raise FileNotFoundError(f"no video found for game {game_id}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start_sec), "-i", str(video),
                 "-t", str(duration_sec), "-c", "copy", str(out_path)],
                check=True, capture_output=True,
            )
        else:
            import cv2
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            n = int(duration_sec * fps)
            for _ in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
            cap.release()
            writer.release()
        return out_path
