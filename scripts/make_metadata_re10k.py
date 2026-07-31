#!/usr/bin/env python3
import argparse, csv, json, os, shutil, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REQUIRED_FIELDS = ["video_path", "caption", "height", "width", "fps", "n_frames"]
EXTRA_FIELDS    = ["split", "pose_path"]  # optional but useful
FIELDNAMES      = REQUIRED_FIELDS + EXTRA_FIELDS

VIDEO_DIRS_DEFAULT = ["training_256", "test_256"]
POSE_DIRS_DEFAULT  = {"training_256": "training_poses", "test_256": "test_poses"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
PREPARED_DIRS = ["training_256", "test_256", "training_poses", "test_poses"]


def has_prepared_layout(root: Path) -> bool:
    return all((root / dirname).is_dir() for dirname in PREPARED_DIRS)


def locate_prepared_layout(root: Path) -> Path:
    candidates = [
        root,
        root / "RealEstate10K_Full" / "real-estate-10k",
        root / "real-estate-10k",
    ]
    for candidate in candidates:
        if has_prepared_layout(candidate):
            return candidate
    expected = ", ".join(PREPARED_DIRS)
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a prepared RE10K layout containing {expected}. "
        f"Searched:\n  {searched}"
    )


def promote_prepared_layout(root: Path) -> None:
    source = locate_prepared_layout(root)
    if source == root:
        print(f"[info] RE10K layout is already prepared under {root}")
        return

    print(f"[info] Promoting prepared RE10K layout from {source} to {root}")
    for dirname in [*PREPARED_DIRS, "metadata"]:
        source_path = source / dirname
        if not source_path.exists():
            if dirname == "metadata":
                continue
            raise FileNotFoundError(f"Missing required RE10K directory: {source_path}")
        destination = root / dirname
        if destination.exists():
            raise FileExistsError(
                f"Refusing to replace existing destination: {destination}"
            )
        try:
            source_path.rename(destination)
        except OSError:
            shutil.move(str(source_path), str(destination))
        print(f"[ok] prepared {destination}")

    if not has_prepared_layout(root):
        raise RuntimeError(f"Failed to prepare RE10K layout under {root}")

def have_ffprobe() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def parse_fraction(fr):
    if not fr or fr == "N/A":
        return None
    if "/" in fr:
        num, den = fr.split("/", 1)
        try:
            return float(num) / float(den)
        except Exception:
            return None
    try:
        return float(fr)
    except Exception:
        return None

def probe_ffprobe(path: Path):
    # Pull stream info (video) + container duration
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames,avg_frame_rate,r_frame_rate",
        "-show_entries","format=duration",
        "-of","json", str(path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {path}")
    j = json.loads(p.stdout)
    stream = (j.get("streams") or [{}])[0]
    fmt = j.get("format") or {}

    width  = stream.get("width")
    height = stream.get("height")

    # fps: prefer avg_frame_rate; fallback to r_frame_rate
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    fps = float(f"{fps:.6f}") if fps else None

    # frames: prefer nb_frames; else duration * fps
    nb_frames = stream.get("nb_frames")
    n_frames = None
    if nb_frames not in (None, "", "N/A", "0", 0):
        try: n_frames = int(nb_frames)
        except: n_frames = None
    if n_frames is None:
        dur = fmt.get("duration")
        if dur not in (None, "", "N/A") and fps:
            try: n_frames = int(round(float(dur) * fps))
            except: n_frames = None

    # Cast ints where possible
    width  = int(width)  if width  is not None else None
    height = int(height) if height is not None else None
    if fps is not None:
        fps = int(round(fps)) if abs(round(fps) - fps) < 1e-3 else float(f"{fps:.3f}")
    return width, height, fps, n_frames

def probe_opencv(path: Path):
    try:
        import cv2
    except ImportError:
        raise RuntimeError("OpenCV not available and ffprobe missing.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    fps    = cap.get(cv2.CAP_PROP_FPS) or None
    nfr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    cap.release()
    if fps is not None:
        fps = int(round(fps)) if abs(round(fps) - fps) < 1e-3 else float(f"{fps:.3f}")
    return width, height, fps, nfr

def find_pose_for_video(root: Path, rel_video: Path, pose_dirs_map: dict) -> str:
    """Match pose by video stem inside the corresponding *poses* dir. Accept .json/.txt or a dir."""
    first = rel_video.parts[0] if rel_video.parts else "" # something like training_256
    poses_dirname = pose_dirs_map.get(first) # can return something like training_poses
    if not poses_dirname:
        return ""
    poses_dir = root / poses_dirname
    if not poses_dir.is_dir():
        return ""

    stem = rel_video.stem
    # exact file hits (common cases)
    for ext in (".json", ".txt", ".npy", ".npz", ".pt"):
        cand = poses_dir / f"{stem}{ext}"
        if cand.exists():
            return str(cand.relative_to(root))
    # or a directory named after stem
    cand_dir = poses_dir / stem
    if cand_dir.is_dir():
        return str(cand_dir.relative_to(root))
    return ""

def enumerate_videos(root: Path, video_dirs: list[str]) -> list[Path]:
    vids = []
    for sub in video_dirs:
        d = root / sub
        if not d.is_dir(): 
            continue
        for p in d.rglob("*"): # gets everything in data/re10k/training_256 or /test_256
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                vids.append(p.relative_to(root)) # gets training_256/filename
    # stable order
    vids.sort()
    return vids

def process_one(root: Path, rel_video: Path, use_ff: bool, pose_dirs_map: dict):
    abs_path = root / rel_video # build path to video like data/re10k/training_256/video.mp4
    # split from top-level dir
    top = rel_video.parts[0] if rel_video.parts else "" # something like training_256
    split = "training" if "train" in top else ("validation" if "test" in top else "")
    # probe
    try:
        width, height, fps, n_frames = (probe_ffprobe(abs_path) if use_ff else probe_opencv(abs_path))
    except Exception as e:
        print(f"[warn] probe failed: {abs_path}: {e}", file=sys.stderr)
        width = height = fps = n_frames = None
    pose_path = find_pose_for_video(root, rel_video, pose_dirs_map)
    return {
        "video_path": str(rel_video),  # relative to root
        "caption": "",                 # fill later if you have captions
        "height": height if height is not None else "",
        "width":  width  if width  is not None else "",
        "fps":    fps    if fps    is not None else "",
        "n_frames": n_frames if n_frames is not None else "",
        "split": split,
        "pose_path": pose_path,
    }

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a downloaded RealEstate10K_Full archive layout and build "
            "the metadata.csv expected by EqF."
        )
    )
    ap.add_argument(
        "--root",
        default="data/re10k",
        help=(
            "Download or dataset root (default: data/re10k). Nested "
            "RealEstate10K_Full/real-estate-10k contents are promoted here."
        ),
    )
    ap.add_argument("--out",  default=None, help="Output CSV path (default: <root>/metadata.csv)")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)), help="Parallel workers")
    ap.add_argument("--video-dirs", nargs="*", default=VIDEO_DIRS_DEFAULT, help="Video subdirs to scan")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    promote_prepared_layout(root)
    out_csv = Path(args.out).resolve() if args.out else (root / "metadata.csv")
    use_ff = have_ffprobe()

    if not use_ff:
        print("[info] ffprobe not found; falling back to OpenCV (slower).", file=sys.stderr)

    vids = enumerate_videos(root, args.video_dirs) # pass in root dir and [†raining_256, test_256]
    if not vids:
        print(f"[error] no videos found under {args.video_dirs} in {root}", file=sys.stderr)
        sys.exit(2)

    pose_dirs_map = {k: v for k, v in POSE_DIRS_DEFAULT.items() if k in args.video_dirs}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, root, rv, use_ff, pose_dirs_map) for rv in vids]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 500 == 0:
                print(f"[progress] {i}/{len(vids)} processed", file=sys.stderr)

    rows.sort(key=lambda r: r["video_path"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ok] wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
