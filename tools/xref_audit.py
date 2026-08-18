"""Cross-reference integrity audit."""
import re
from pathlib import Path

paths = [
    ("ICLR", r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper\main.tex"),
    ("TMLR", r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper_tmlr\main.tex"),
]

for label, p in paths:
    text = Path(p).read_text(encoding='utf-8')
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs   = set(re.findall(r"\\ref\{([^}]+)\}", text))
    missing = sorted(refs - labels)
    print(f"\n{label}: {len(labels)} labels, {len(refs)} refs, missing: {len(missing)}")
    for m in missing[:30]:
        print(f"    MISSING: {m}")
    unused = sorted(labels - refs)
    print(f"  unused labels: {len(unused)}")
    for u in unused[:10]:
        print(f"    unused: {u}")
