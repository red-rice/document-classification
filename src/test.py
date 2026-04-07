from pathlib import Path

project_root = Path(r"F:\Project")
test_file = project_root / "test.txt"
box_root = project_root / "rvl-cdip-box"
img_root = project_root / "rvl-cdip"

count_total = 0
count_ok = 0

with test_file.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 500:   # same as debug_samples
            break
        rel = line.strip().split()[0]
        img = img_root / rel
        box = box_root / Path(rel).with_suffix(".tsv")
        count_total += 1
        if img.exists() and box.exists():
            count_ok += 1

print("test samples checked:", count_total)
print("samples with both image and box:", count_ok)