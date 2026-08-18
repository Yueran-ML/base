"""Format compliance audit for ICLR + TMLR papers."""
from pathlib import Path
import re

paths = [
    ("ICLR", r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper\main.tex"),
    ("TMLR", r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper_tmlr\main.tex"),
]

for label, path in paths:
    text = Path(path).read_text(encoding='utf-8')
    print(f"\n=== {label}: {path} ===")
    print(f"  total lines: {len(text.splitlines())}")

    n_cite  = len(re.findall(r'\\cite\{', text))
    n_citet = len(re.findall(r'\\citet\{', text))
    n_citep = len(re.findall(r'\\citep\{', text))
    print(f"  \\cite{{}}: {n_cite}   \\citet: {n_citet}   \\citep: {n_citep}")

    dup = re.findall(
        r"[A-Z][a-zA-Z']+(?: et al\.| and [A-Z][a-zA-Z]+)?~\\cite\{[^}]+\}",
        text,
    )
    print(f"  surviving 'Author~\\cite' patterns: {len(dup)}")

    print(f"  uses iclr2026_conference: {'iclr2026_conference' in text}")
    print(f"  uses tmlr.sty: {'usepackage{tmlr}' in text}")
    print(f"  has Ethics Statement: {'Ethics Statement' in text}")
    print(f"  has Reproducibility Statement: {'Reproducibility Statement' in text}")
    print(f"  has LLM Usage Statement: {'LLM Usage Statement' in text}")
    print(f"  has Broader Impact: {'Broader Impact' in text}")

    worklog = re.findall(
        r"has (since |now )?been (executed|run|completed)",
        text,
        re.IGNORECASE,
    )
    print(f"  work-log 'has been executed/run/completed': {len(worklog)}")
    todos = len(re.findall(r"\bTODO\b", text))
    print(f"  TODO markers: {todos}")

    # Section spans for main body length sanity check
    sections = []
    for m in re.finditer(r"^\\section\{([^}]+)\}", text, re.MULTILINE):
        sections.append((m.start(), m.group(1)))
    if sections:
        ln_at = lambda pos: text[:pos].count("\n") + 1
        print("  section line numbers:")
        for pos, name in sections[:12]:
            print(f"    {ln_at(pos):>5}  {name}")
        # estimate main body length: from line 1 to start of first appendix-y section
        main_end = None
        for pos, name in sections:
            low = name.lower()
            if any(k in low for k in ["experimental details", "background (extended)",
                                     "extended robustness", "complete timing",
                                     "per-cell detector", "metrics: extended",
                                     "speed-dependent hypothesis: full"]):
                main_end = ln_at(pos)
                break
        if main_end:
            print(f"  estimated main body lines (up to first appendix section): {main_end}")
