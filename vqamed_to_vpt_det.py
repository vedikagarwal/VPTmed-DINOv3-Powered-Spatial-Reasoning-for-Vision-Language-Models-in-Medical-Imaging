#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA-Med 2019 -> VPT-Det JSONL (directly emits wrapped detection-action format)

- Never writes dataset_info.json.
- Writes *.debug.txt ONLY when --debug is supplied.

messages[0] (user):
  "<image>\n{question}\nRequire additional perception features, and then answer the question.\n<detection_image>"
messages[1] (assistant):
  "<|detection_action_start|>\n<|detection_action|>{\"boxes\": [], \"labels\": []}\n<|detection_action_end|>\n{answer_or_empty}"
detection_images: [resolved image path]
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

DA_START = "<|detection_action_start|>"
DA       = "<|detection_action|>"
DA_END   = "<|detection_action_end|>"

def norm_ans(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def read_text(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        return [ln.rstrip("\r\n") for ln in f]

def sniff_delim_by_counts(lines: List[str]) -> str:
    cand = ["|", "\t", ",", ";"]
    counts = {c: 0 for c in cand}
    n = 0
    for ln in lines:
        if not ln.strip():
            continue
        n += 1
        for c in cand:
            counts[c] += ln.count(c)
        if n >= 200:
            break
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else "\t"

def split_flexible(ln: str, delim: str) -> List[str]:
    if delim in {"|", "\t", ",", ";"}:
        return [c.strip() for c in ln.split(delim)]
    return [c.strip() for c in re.split(r"\s{2,}", ln)]

def try_csv_reader(path: Path, delim: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        rdr = csv.reader(f, delimiter=("\t" if delim == "\t" else ("," if delim == "," else delim)))
        for r in rdr:
            rows.append([c.strip() for c in r])
    return rows

def read_qa_table(qa_file: Path) -> List[Dict[str, str]]:
    lines = read_text(qa_file)
    if not lines:
        return []
    delim = sniff_delim_by_counts(lines)

    raw = try_csv_reader(qa_file, delim)
    if delim == "|" and raw and max(len(r) for r in raw) <= 1:
        raw = [split_flexible(ln, "|") for ln in lines]

    header = [c.lower() for c in raw[0]] if raw else []
    data = raw[1:] if header and any(h in {"image","imageid","image_id","imagename","file","filename","question"} for h in header) else raw

    def col_idx(names: List[str], *aliases) -> Optional[int]:
        for a in aliases:
            if a in names:
                return names.index(a)
        return None

    rows: List[Dict[str, str]] = []

    if header and data is not raw:
        img_i = col_idx(header, "image","imageid","image_id","imagename","file","filename","img","img_id")
        q_i   = col_idx(header, "question","ques","q","questiontext","question_text")
        a_i   = col_idx(header, "answer","ans","a","groundtruth","gt")
        for r in data:
            if not r:
                continue
            image = (r[img_i] if img_i is not None and img_i < len(r) else "").strip()
            ques  = (r[q_i]   if q_i   is not None and q_i   < len(r) else "").strip()
            ans   = (r[a_i]   if a_i   is not None and a_i   < len(r) else "").strip()
            if image and ques:
                rows.append({"image": image, "question": ques, "answer": ans})
        if rows:
            return rows

    for r in raw:
        if not r:
            continue
        parts = [p for p in r if p != ""]
        if len(parts) < 2:
            parts = split_flexible(" ".join(r), delim)
            if len(parts) < 2:
                continue
        image, ques = parts[0].strip(), parts[1].strip()
        ans = parts[2].strip() if len(parts) >= 3 else ""
        if image and ques:
            rows.append({"image": image, "question": ques, "answer": ans})
    return rows

def find_first_dir(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for c in candidates:
        p = (root / c).resolve()
        if p.exists() and p.is_dir():
            return p
    cand_lower = {c.lower() for c in candidates}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in cand_lower:
            return p
    return None

def find_images_dir(root: Path, explicit: Optional[str], split_hint: str) -> Path:
    if explicit:
        p = (root / explicit).resolve()
        if p.exists() and p.is_dir():
            return p
    common = {
        "train": ["Train_images","TrainImages","Images","images"],
        "val":   ["Val_images","ValidationImages","ValImages","Images","images"],
        "test":  ["Test_Images","VQAMed2019_Test_Images","TestImages","Images","images"],
    }
    candidates = common.get(split_hint, []) + ["Images","images"]
    p = find_first_dir(root, candidates)
    return (p or root).resolve()

def build_image_index(img_root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    stems: Dict[str, Optional[Path]] = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[p.name.lower()] = p.resolve()
            st = p.stem.lower()
            stems[st] = p.resolve() if st not in stems else None
    stems = {k: v for k, v in stems.items() if v is not None}
    idx.update({f"STEM::{k}": v for k, v in stems.items()})  # type: ignore
    return idx

def resolve_image(image_name: str, image_index: Dict[str, Path]) -> Optional[Path]:
    base = Path(image_name.strip()).name.lower()
    if base in image_index:
        return image_index[base]
    stem = Path(base).stem.lower()
    return image_index.get(f"STEM::{stem}", None)

def make_wrapped_record(img_path: Path, question: str, answer: Optional[str]) -> Dict:
    user_q = (
        "<image>\n"
        f"{question}\n"
        "Require additional perception features, and then answer the question.\n"
        "<detection_image>"
    )
    ans = "" if answer is None else norm_ans(answer)
    assistant_content = (
        f"{DA_START}\n"
        f'{DA}{{"boxes": [], "labels": []}}\n'
        f"{DA_END}\n"
        f"{ans}"
    )
    return {
        "messages": [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_content},
        ],
        "detection_images": [str(img_path)],
    }

@dataclass
class SplitSpec:
    root: Path
    qa_file: Optional[Path]
    images_subdir: Optional[str]
    split_hint: str
    is_test: bool

def find_qa_file(root: Path) -> Optional[Path]:
    patterns = [
        "All_QA*train*", "All_QA*val*", "All_QA*validation*",
        "*qapairs*.txt", "*qapair*.txt", "*questions*.txt",
        "*train*.csv","*train*.tsv","*train*.txt",
        "*val*.csv","*val*.tsv","*val*.txt",
        "*validation*.csv","*validation*.tsv","*validation*.txt",
        "VQAMed2019_Test_Questions*"
    ]
    for pat in patterns:
        for f in root.rglob(pat):
            if f.is_file() and f.suffix.lower() in {".csv",".tsv",".txt"}:
                return f.resolve()
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".csv",".tsv",".txt"}:
            return f.resolve()
    return None

def convert_split(spec: SplitSpec, out_jsonl: Path, debug: bool=False, tag: str="") -> Tuple[int,int,int]:
    img_dir = find_images_dir(spec.root, spec.images_subdir, spec.split_hint)
    image_index = build_image_index(img_dir)

    qa_file = spec.qa_file or find_qa_file(spec.root)
    if qa_file is None:
        raise FileNotFoundError(f"Could not find QA file under {spec.root}; specify --{spec.split_hint}-qa.")
    rows = read_qa_table(qa_file)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_resolved = 0
    n_written = 0
    unresolved_examples: List[str] = []

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            img_rel = r.get("image","").strip()
            q = r.get("question","").strip()
            a = r.get("answer","").strip()
            if not img_rel or not q:
                continue
            img_path = resolve_image(img_rel, image_index)
            if img_path is None:
                if debug and len(unresolved_examples) < 20:
                    unresolved_examples.append(img_rel)
                continue
            n_resolved += 1
            rec = make_wrapped_record(img_path, q, (None if spec.is_test else a))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    # Only write a debug file when --debug is set
    if debug:
        dbg = out_jsonl.with_suffix(".debug.txt")
        with dbg.open("w", encoding="utf-8") as f:
            print(f"[{tag}] qa_file={qa_file}", file=f)
            print(f"[{tag}] images_dir={img_dir}", file=f)
            print(f"[{tag}] rows={len(rows)} resolved={n_resolved} written={n_written}", file=f)
            if unresolved_examples:
                print(f"[{tag}] unresolved (first {len(unresolved_examples)}):", file=f)
                for u in unresolved_examples:
                    print("  ", u, file=f)
    return (len(rows), n_resolved, n_written)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--val-root",   type=Path, required=True)
    ap.add_argument("--test-root",  type=Path, required=True)
    ap.add_argument("--train-qa",   type=Path, default=None)
    ap.add_argument("--val-qa",     type=Path, default=None)
    ap.add_argument("--test-qa",    type=Path, default=None)
    ap.add_argument("--train-images-subdir", type=str, default=None)
    ap.add_argument("--val-images-subdir",   type=str, default=None)
    ap.add_argument("--test-images-subdir",  type=str, default=None)
    ap.add_argument("--out-dir",    type=Path, required=True)
    ap.add_argument("--prefix",     type=str, default="vqamed2019_vpt_det")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = SplitSpec(args.train_root.resolve(), args.train_qa.resolve() if args.train_qa else None,
                   args.train_images_subdir, "train", False)
    va = SplitSpec(args.val_root.resolve(),   args.val_qa.resolve() if args.val_qa else None,
                   args.val_images_subdir,   "val",   False)
    te = SplitSpec(args.test_root.resolve(),  args.test_qa.resolve() if args.test_qa else None,
                   args.test_images_subdir,  "test",  True)

    tr_out = out_dir / f"{args.prefix}_train.jsonl"
    va_out = out_dir / f"{args.prefix}_val.jsonl"
    te_out = out_dir / f"{args.prefix}_test.jsonl"

    tr_rows, tr_res, tr_written = convert_split(tr, tr_out, args.debug, "Train")
    print(f"[Train] rows={tr_rows} resolved={tr_res} written={tr_written} -> {tr_out}")
    va_rows, va_res, va_written = convert_split(va, va_out, args.debug, "Val")
    print(f"[Val]   rows={va_rows} resolved={va_res} written={va_written} -> {va_out}")
    te_rows, te_res, te_written = convert_split(te, te_out, args.debug, "Test")
    print(f"[Test]  rows={te_rows} resolved={te_res} written={te_written} -> {te_out}")

    for name, rows, res in [("Train", tr_rows, tr_res), ("Val", va_rows, va_res), ("Test", te_rows, te_res)]:
        if rows > 0 and res == 0:
            print(f"[Hint] {name}: 0/{rows} resolved. Check delimiter or filename ID conventions.",
                  file=sys.stderr)

if __name__ == "__main__":
    main()
