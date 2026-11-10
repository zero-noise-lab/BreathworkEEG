"""Walk the ZNLRESET transcript tree and emit a tidy CSV table.

Expected source layout::

    <data-root>/<person_id>/<session_type>/video_*/transcripts/transcripts_c*/*.txt

Each ``.txt`` file is a single Whisper chunk with a header of the form::

    [VIDEO TIMESTAMP: HH:MM:SS.fff][QUALITY WARNINGS: silence_detected]Body...

Usage
-----
    python load_transcripts.py --session-type BW \\
        --output data/transcripts_BW.csv

    python load_transcripts.py --session-type all --filter-quality
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from transcribe_utils import ParsedChunk, parse_chunk_file

HPC_DATA_ROOT = "" # Prjectroot
SESSION_TYPES = ("BW", "PMR")
SESSION_CHOICES = (*SESSION_TYPES, "all")
DEFAULT_FPS = 50  # Video acquisition FPS used to compute `frame` from timestamp_s

# Filenames inside transcripts/transcripts/ that are NOT Whisper chunks and
# should be skipped silently (without triggering a parse-error warning).
EXCLUDE_FILENAMES: frozenset[str] = frozenset({"temp_audio.txt"})

logger = logging.getLogger("load_transcripts")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path, default=None,
                     help="Root containing per-person subfolders — alias for --data-root, "
                          "intended for ad-hoc test runs (e.g. a copy of one VP to /scratch).")
    src.add_argument("--data-root", type=Path, default=None,
                     help=f"Canonical HPC root (default if neither --input nor --data-root is given: "
                          f"{HPC_DATA_ROOT})")
    p.add_argument("--session-type", required=True, choices=SESSION_CHOICES,
                   help="Which session_type to include")
    p.add_argument("--person-id", action="append", default=None,
                   help="Restrict to one or more person IDs (Chiffre). Repeat flag for multiples.")
    p.add_argument("--filter-quality", action="store_true",
                   help="Drop chunks that carry any [QUALITY WARNINGS: ...] tag")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"Video FPS used to derive the `frame` column from timestamp_s "
                        f"(default: {DEFAULT_FPS})")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default: ./data/transcripts_<session>.csv)")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)
    # Resolve effective data root: --input > --data-root > HPC default
    args.data_root = args.input or args.data_root or Path(HPC_DATA_ROOT)
    return args


def iter_chunk_files(data_root: Path, session_type: str,
                     person_ids: list[str] | None) -> list[Path]:
    """Enumerate chunk ``.txt`` files matching the requested filters."""
    if not data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {data_root}")

    sessions = SESSION_TYPES if session_type == "all" else (session_type,)
    persons = person_ids if person_ids else sorted(p.name for p in data_root.iterdir() if p.is_dir())

    files: list[Path] = []
    for person in persons:
        person_dir = data_root / person
        if not person_dir.is_dir():
            logger.warning("Person dir missing, skipping: %s", person_dir)
            continue
        for session in sessions:
            session_dir = person_dir / session
            if not session_dir.is_dir():
                logger.debug("No %s session for person %s", session, person)
                continue
            for video_dir in sorted(session_dir.glob("video_*")):
                transcripts_dir = video_dir / "transcripts"
                if not transcripts_dir.is_dir():
                    logger.debug("Missing transcripts dir: %s", transcripts_dir)
                    continue
                txt_dirs = sorted(transcripts_dir.glob("transcripts_c*"))
                if not txt_dirs:
                    logger.debug("No transcripts_c* subdirs in: %s", transcripts_dir)
                    continue
                for txt_dir in txt_dirs:
                    files.extend(
                        p for p in sorted(txt_dir.glob("*.txt"))
                        if p.name not in EXCLUDE_FILENAMES
                    )
    return files


def _chunk_to_row(parsed: ParsedChunk, data_root: Path) -> dict:
    """Extract (person_id, session_type, video_id) from the chunk path."""
    rel = parsed.path.relative_to(data_root)
    # rel = <person>/<session>/<video_*>/transcripts/transcripts/<file>.txt
    parts = rel.parts
    person_id = parts[0]
    session_type = parts[1]
    video_id = parts[2]
    chunk_filename = parts[-1]
    n_words_raw = len(parsed.text.split())
    return {
        "person_id": person_id,
        "session_type": session_type,
        "video_id": video_id,
        "chunk_filename": chunk_filename,
        "timestamp_s": parsed.timestamp_s,
        "timestamp_str": parsed.timestamp_str,
        "quality_warnings": ",".join(parsed.quality_warnings),
        "filter_tag": parsed.filter_tag or "",
        "text": parsed.text,
        "n_chars": len(parsed.text),
        "n_words_raw": n_words_raw,
    }


def build_dataframe(files: list[Path], data_root: Path,
                    filter_quality: bool, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Parse all chunk files into a tidy DataFrame ordered by timestamp.

    Adds a ``frame`` column derived from ``round(timestamp_s * fps)`` so chunk
    rows can be joined to per-frame facial-landmark / blendshape tables.
    """
    rows: list[dict] = []
    parse_errors = 0
    for path in files:
        try:
            parsed = parse_chunk_file(path)
        except ValueError as e:
            logger.warning("Parse error %s — skipping", e)
            parse_errors += 1
            continue
        if filter_quality and parsed.quality_warnings:
            continue
        rows.append(_chunk_to_row(parsed, data_root))

    if parse_errors:
        logger.warning("Total parse errors: %d (of %d files)", parse_errors, len(files))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["person_id", "session_type", "video_id", "timestamp_s"]).reset_index(drop=True)
    df["chunk_idx"] = (
        df.groupby(["person_id", "session_type", "video_id"]).cumcount()
    )
    df["frame"] = (df["timestamp_s"] * fps).round().astype("int64")
    return df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    output = args.output
    if output is None:
        Path("data").mkdir(exist_ok=True)
        output = Path(f"data/transcripts_{args.session_type}.csv")

    files = iter_chunk_files(args.data_root, args.session_type, args.person_id)
    logger.info("Discovered %d chunk files", len(files))
    if not files:
        logger.error("No chunk files matched the filters — nothing to write.")
        return 1

    df = build_dataframe(files, args.data_root, args.filter_quality, fps=args.fps)
    logger.info("Parsed %d chunks (after filter_quality=%s, fps=%g)",
                len(df), args.filter_quality, args.fps)
    if df.empty:
        logger.error("DataFrame is empty after parsing/filters.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote %s (%d rows, persons=%d, sessions=%s)",
                output, len(df), df["person_id"].nunique(),
                sorted(df["session_type"].unique()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
