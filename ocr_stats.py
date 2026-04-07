import os
from pathlib import Path

def check_ocr_stats(root_dir, suffix, limit=1000):
    root = Path(root_dir)
    files = list(root.rglob(f"*{suffix}"))[:limit]
    
    total = len(files)
    empty = 0
    very_short = 0
    
    for f in files:
        content = f.read_text(encoding='utf-8', errors='ignore').strip()
        if not content:
            empty += 1
        elif len(content.split()) < 5:
            very_short += 1
            
    print(f"\nStats for {root_dir} ({suffix}):")
    print(f"Total checked: {total}")
    print(f"Empty: {empty} ({empty/total:.1%})")
    print(f"Very short (<5 words): {very_short} ({very_short/total:.1%})")

check_ocr_stats("rvl-cdip-text", ".txt")
check_ocr_stats("rvl-cdip-box", ".tsv")
