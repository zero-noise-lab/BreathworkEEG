#!/usr/bin/env python3
"""Synthetic stand-ins for the word-cloud pipeline. No real data, fixed seed.

Usage:
  python make_synthetic_data.py                        # built-in phase template
  python make_synthetic_data.py --template-from <real phase_intervals.csv>
Then run the notebook with:  RESET_DATA_DIR=input/synthetic/data_annotations
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED, FPS = 42, 50
CHUNKS_PER_PERSON = (10, 15)                       # inclusive range
IDS = [f"SYN{i:03d}" for i in range(1, 26)]        # 25 = expected_vps in Zelle 10
EXCLUDED = [f"SYX{i:03d}" for i in range(1, 6)]    # = EXCLUDED_VPS in Zelle 1
PERSONS = IDS + EXCLUDED
TYPO = {"SYN025": "SYNO25"}                        # Rohschreibweise mit Buchstabe O

CONTENT = """warmth tingling pressure chest hands light heavy floating dissolving peace calm
energy waves vibration tension release opening fear joy gratitude silence space unity
colors patterns memory childhood tears laughter heartbeat stillness expansion boundary
safe lost clear present surrender trembling pulse stomach throat shoulders dizzy
connected spacious soft sharp bright dark deep slow fast rhythm""".split()
FILLER = """i the a and it was like yeah really um know think kind of bit okay so that
this in to my me felt just then there is were with maybe actually sort of a lot
breath body feel session""".split()


def phase_template(path: Path | None) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)[["phase", "start_frame", "end_frame", "max_frame"]]
        return df.sort_values("start_frame").reset_index(drop=True)
    names = (["EyesOpen1"] + [f"{k}{i}" for i in range(1, 5) for k in ("Resting", "Breathing")]
             + ["Resting5", "EyesOpen2", "Post1", "Pre1"])
    names = ["Pre1", "EyesOpen1"] + [f"{k}{i}" for i in range(1, 5) for k in ("Resting", "Breathing")] \
            + ["Resting5", "EyesOpen2", "Post1"]          # 13 phases
    step = 19_500
    return pd.DataFrame({"phase": names,
                         "start_frame": [i * step for i in range(len(names))],
                         "end_frame": [(i + 1) * step - 1 for i in range(len(names))],
                         "max_frame": len(names) * step})


def make_text(rng) -> str:
    n = int(rng.integers(8, 90))
    words = [rng.choice(CONTENT) if rng.random() < 0.4 else rng.choice(FILLER) for _ in range(n)]
    return (" ".join(words).capitalize() + ".")


def make_transcripts(rng, tpl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p_i, pid in enumerate(PERSONS):
        n = int(rng.integers(CHUNKS_PER_PERSON[0], CHUNKS_PER_PERSON[1] + 1))
        picks = tpl.iloc[rng.integers(0, len(tpl), n)]
        frames = sorted({int(rng.integers(r.start_frame, r.end_frame)) for r in picks.itertuples()})
        stamp = f"2025-0{1 + p_i % 9}-1{p_i % 10}T10_{10 + p_i}_00"
        for idx, fr in enumerate(frames):
            t = fr / FPS
            text = "" if rng.random() < 0.05 else make_text(rng)
            qw = rng.choice([np.nan, "silence_detected", "timeless", "silence_detected,timeless"],
                            p=[0.91, 0.04, 0.03, 0.02])
            ft = np.nan if rng.random() < 0.23 else rng.choice(
                ["experience", "experience_affect", "experience_body", "experience_insight"])
            h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
            rows.append({
                "person_id": pid, "session_type": "BW",
                "video_id": f"Synth_Recording_{stamp}",
                "chunk_filename": f"clip_{fr:08d}_{int(t):06d}_{int(t) + 20:06d}.wav",
                "timestamp_s": round(t, 3),
                "timestamp_str": f"{h:02d}:{m:02d}:{s:06.3f}",
                "quality_warnings": qw, "filter_tag": ft, "text": text,
                "n_chars": len(text), "n_words_raw": len(text.split()),
                "chunk_idx": idx, "frame": fr})
    return pd.DataFrame(rows)


def make_asc(rng) -> pd.DataFrame:
    vas_max = {"unity": 75, "spiritual": 67, "blissful": 66, "insight": 99, "disembodiment": 98,
               "impairedcontrol": 71, "anxiety": 88, "complexim": 66, "elemin": 85, "syn": 65,
               "changedmean": 71, "OBN": 74, "DED": 80, "VR": 66}
    vas_na = {"complexim", "elemin", "syn", "VR"}
    t_max = [2, 3, 4, 4, 5, 5, 5, 5, 5, 5]
    plan = [(p, "BW") for p in PERSONS] + [(p, "PMR") for p in PERSONS[::2]] + [(PERSONS[1], "BW")]
    rows = []
    for k, (pid, sess) in enumerate(plan):
        sub = rng.uniform(0.05, 0.95, 4).round(3)          # trans, pos, inef, mys
        d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.choice([2024, 2025]))
        row = {"subj_num": 6 + PERSONS.index(pid), "session": sess, "subject": pid,
               "meq_trans": sub[0], "meq_pos": sub[1], "meq_inef": sub[2], "meq_mys": sub[3],
               "meq_total": round(float(sub.mean()), 4)}
        for c, hi in vas_max.items():
            row[c] = np.nan if (c in vas_na and rng.random() < 0.04) else round(float(rng.uniform(0, hi)), 1)
        row |= {"timepoint": str(rng.choice(["t1_1_arm_1", "t1_2_arm_1"])), "ParticipantID": pid,
                "SessionOrder": int(rng.integers(1, 4)), "SessionDateDay": d, "SessionDateMonth": m,
                "SessionDateYear": y, "SessionDate": f"{d:02d}/{m:02d}/{y}", "SessionType": sess,
                "t1": int(rng.integers(1, 3))}
        for i, hi in enumerate(t_max[1:], start=2):
            row[f"t{i}"] = float(rng.integers(1, hi + 1))
        for i in range(1, 11):
            row[f"t{i}_meq"] = round(float(rng.uniform(1, 4.5)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="input/synthetic/data_annotations")
    ap.add_argument("--template-from", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "transcripts_bw.csv").exists() and not a.force:
        raise SystemExit(f"{out} already contains data; use --force to overwrite.")
    rng = np.random.default_rng(SEED)
    tpl = phase_template(a.template_from)

    out.mkdir(parents=True, exist_ok=True)
    tr = make_transcripts(rng, tpl)
    tr.to_csv(out / "transcripts_bw.csv", index=False)
    make_asc(rng).to_csv(out / "ASC_all.csv", index=False)
    for pid in PERSONS:
        d = out / "session_phases" / pid
        d.mkdir(parents=True, exist_ok=True)
        t = tpl.copy()
        t.insert(0, "face_id", pid)
        t.to_csv(d / "phase_intervals.csv", index=False)
    (out / "vp_config.json").write_text(json.dumps(
        {"excluded_vps": EXCLUDED, "face_id_corrections": {}, "expected_vps": IDS}, indent=2))
    print(f"{len(tr)} chunks, {len(PERSONS)} persons -> {out}")


if __name__ == "__main__":
    main()