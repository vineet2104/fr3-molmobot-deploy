"""Stitch exo (left) + wrist (right) into a rollout video with the TASK
instruction written on a top banner. H.264 / yuv420p mp4 (plays in any Linux
player: VLC, mpv, browsers, GNOME Videos).

Usage:
    python tools/stitch_task_videos.py --pattern "logs/20260817-*" [--overwrite] [--out rollout.mp4]

Task text is read from <exp>/run_meta.json -> args.task.
FPS from run_meta args.rate (default 15).
"""
from __future__ import annotations
import argparse, glob, json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np

FFMPEG = (shutil.which("ffmpeg") or "/home/franka/miniforge3/bin/ffmpeg")


def _task_and_fps(exp: Path):
    task, fps = "", 15
    mp = exp / "run_meta.json"
    if mp.exists():
        try:
            m = json.loads(mp.read_text())
            a = m.get("args", {}) or {}
            task = a.get("task") or a.get("prompt") or ""
            fps = int(round(float(a.get("rate", 15) or 15)))
        except Exception:
            pass
    return task, max(1, fps)


def _label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def _banner(width, task, height=46):
    """Black top banner with the task centered; auto-fit font to width."""
    bar = np.zeros((height, width, 3), np.uint8)
    text = f"Task: {task}" if task else "Task: (unknown)"
    scale = 0.9
    thick = 2
    while scale > 0.35:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= width - 20:
            break
        scale -= 0.05
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(8, (width - tw) // 2)
    y = (height + th) // 2
    cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return bar


def stitch(exp: Path, out_name: str, overwrite: bool):
    exo_dir, wr_dir = exp / "frames", exp / "frames_wrist"
    if not exo_dir.is_dir():
        return False, "no frames/ (exo)"
    exo = sorted(glob.glob(str(exo_dir / "*.jpg")))
    wr = sorted(glob.glob(str(wr_dir / "*.jpg"))) if wr_dir.is_dir() else []
    if not exo:
        return False, "no exo frames"
    out = exp / out_name
    if out.exists() and not overwrite:
        return False, "exists (use --overwrite)"
    have_wrist = bool(wr)
    n = min(len(exo), len(wr)) if have_wrist else len(exo)
    task, fps = _task_and_fps(exp)

    im0 = cv2.imread(exo[0])
    if im0 is None:
        return False, f"cannot read {exo[0]}"
    h, w = im0.shape[:2]
    W = w * 2 if have_wrist else w
    bh = 46
    H = h + bh
    banner = _banner(W, task, bh)

    ff = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(n):
        e = cv2.imread(exo[i])
        if e is None:
            continue
        if e.shape[:2] != (h, w):
            e = cv2.resize(e, (w, h))
        panes = [_label(e, "exo")]
        if have_wrist:
            wimg = cv2.imread(wr[i])
            wimg = np.zeros((h, w, 3), np.uint8) if wimg is None else (
                wimg if wimg.shape[:2] == (h, w) else cv2.resize(wimg, (w, h)))
            panes.append(_label(wimg, "wrist"))
        body = cv2.hconcat(panes) if len(panes) > 1 else panes[0]
        frame = np.ascontiguousarray(np.vstack([banner, body]))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close(); ff.wait()
    return True, f"{n} frames @ {fps}fps, task={task!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help='e.g. "logs/20260817-*"')
    ap.add_argument("--out", default="rollout.mp4")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    dirs = sorted(p for p in map(Path, glob.glob(args.pattern)) if p.is_dir())
    print(f"stitching {len(dirs)} experiment(s)")
    okc = 0
    for d in dirs:
        ok, msg = stitch(d, args.out, args.overwrite)
        print(f"  {'OK  ' if ok else 'skip'}  {d.name:44s} {msg}")
        okc += ok
    print(f"done: {okc}/{len(dirs)} written")


if __name__ == "__main__":
    main()
