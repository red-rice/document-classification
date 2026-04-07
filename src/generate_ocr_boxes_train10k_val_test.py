import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pytesseract
from PIL import Image, ImageSequence
from tqdm import tqdm


def parse_split_line(line: str) -> Optional[Tuple[str, int]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Bad split line: {line!r}")
    rel_path = parts[0].replace("\\", "/").lstrip("./")
    label = int(parts[1])
    return rel_path, label


def collect_split_paths(project_root: Path, split_name: str, limit: Optional[int] = None) -> List[str]:
    split_path = project_root / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    items = []
    with split_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_split_line(line)
            if parsed is None:
                continue
            rel_path, _ = parsed
            items.append(rel_path)
            if limit is not None and len(items) >= limit:
                break
    return items


def iter_tiff_pages(img_path: Path, max_pages: Optional[int] = None):
    with Image.open(img_path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if max_pages is not None and i >= max_pages:
                break
            yield frame.convert("RGB")


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def extract_word_boxes_from_page(page_image: Image.Image, lang: str, psm: int, oem: int):
    config = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(
        page_image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    n = len(data["text"])
    rows = []

    for i in range(n):
        text = data["text"][i].strip()
        conf = safe_int(data["conf"][i], default=-1)

        if not text:
            continue
        if conf < 0:
            continue

        x = safe_int(data["left"][i])
        y = safe_int(data["top"][i])
        w = safe_int(data["width"][i])
        h = safe_int(data["height"][i])

        if w <= 0 or h <= 0:
            continue

        rows.append((text, x, y, x + w, y + h))

    return rows


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_tsv(tsv_path: Path, rows: List[Tuple[str, int, int, int, int]]):
    ensure_parent(tsv_path)
    with tsv_path.open("w", encoding="utf-8", errors="ignore") as f:
        f.write("word\tx1\ty1\tx2\ty2\n")
        for word, x1, y1, x2, y2 in rows:
            word = word.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            if not word:
                continue
            f.write(f"{word}\t{x1}\t{y1}\t{x2}\t{y2}\n")


def dedupe_keep_order(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--rvl_dir", type=str, default="rvl-cdip")
    ap.add_argument("--out_dir", type=str, default="rvl-cdip-box")

    # split-specific debug limits
    ap.add_argument("--train_limit", type=int, default=30000)
    ap.add_argument("--val_limit", type=int, default=5000)
    ap.add_argument("--test_limit", type=int, default=5000)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--missing_only", action="store_true")
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--oem", type=int, default=3)
    ap.add_argument("--max_pages", type=int, default=1)
    ap.add_argument("--tesseract_cmd", type=str, default=None)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    rvl_root = project_root / args.rvl_dir
    out_root = project_root / args.out_dir

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    train_paths = collect_split_paths(project_root, "train.txt", limit=args.train_limit)
    val_paths = collect_split_paths(project_root, "val.txt", limit=args.val_limit)
    test_paths = collect_split_paths(project_root, "test.txt", limit=args.test_limit)

    rel_paths = dedupe_keep_order(train_paths + val_paths + test_paths)

    print(f"Train selected: {len(train_paths)}")
    print(f"Val selected:   {len(val_paths)}")
    print(f"Test selected:  {len(test_paths)}")
    print(f"Total unique:   {len(rel_paths)}")
    print(f"RVL root:       {rvl_root}")
    print(f"Output root:    {out_root}")

    to_process = []
    missing_images = 0
    existing_tsv = 0

    for rel_path in rel_paths:
        img_path = rvl_root / rel_path
        tsv_path = (out_root / Path(rel_path)).with_suffix(".tsv")

        if not img_path.exists():
            missing_images += 1
            continue

        if tsv_path.exists():
            existing_tsv += 1
            if args.missing_only and not args.overwrite:
                continue
            if (not args.overwrite) and (not args.missing_only):
                continue

        to_process.append((img_path, tsv_path))

    print(f"Missing images: {missing_images}")
    print(f"Existing TSVs:  {existing_tsv}")
    print(f"Will process:   {len(to_process)}")

    processed = 0
    errors = 0
    empty_docs = 0

    for img_path, tsv_path in tqdm(to_process, desc="Generating OCR boxes"):
        try:
            all_rows = []

            for page in iter_tiff_pages(img_path, max_pages=args.max_pages):
                rows = extract_word_boxes_from_page(
                    page_image=page,
                    lang=args.lang,
                    psm=args.psm,
                    oem=args.oem,
                )
                all_rows.extend(rows)

            if len(all_rows) == 0:
                empty_docs += 1

            write_tsv(tsv_path, all_rows)
            processed += 1

        except Exception as e:
            errors += 1
            ensure_parent(tsv_path)
            with tsv_path.open("w", encoding="utf-8", errors="ignore") as f:
                f.write("word\tx1\ty1\tx2\ty2\n")
            print(f"\n[ERROR] {img_path}: {repr(e)}")

    print("\nDone.")
    print(f"Processed:     {processed}")
    print(f"Errors:        {errors}")
    print(f"Empty docs:    {empty_docs}")
    print(f"Output folder: {out_root}")


if __name__ == "__main__":
    main()