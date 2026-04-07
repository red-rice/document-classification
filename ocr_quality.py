import os
from pathlib import Path
import statistics

def check_ocr_quality(root_dir, suffix, limit=1000):
    root = Path(root_dir)
    files = list(root.rglob(f"*{suffix}"))[:limit]
    
    num_words = []
    avg_word_len = []
    
    for f in files:
        if suffix == ".tsv":
            with open(f, "r", encoding='utf-8', errors='ignore') as f_obj:
                lines = f_obj.readlines()
                words = [ln.split("\t")[0] for ln in lines[1:]]
        else:
            content = f.read_text(encoding='utf-8', errors='ignore').strip()
            words = content.split()
            
        if not words:
            continue
            
        num_words.append(len(words))
        avg_word_len.append(statistics.mean([len(w) for w in words]))
            
    print(f"\nStats for {root_dir}:")
    print(f"Mean words per doc: {statistics.mean(num_words):.1f}")
    print(f"Median words per doc: {statistics.median(num_words):.1f}")
    print(f"Mean word length: {statistics.mean(avg_word_len):.1f}")

check_ocr_quality("rvl-cdip-text", ".txt")
check_ocr_quality("rvl-cdip-box", ".tsv")
