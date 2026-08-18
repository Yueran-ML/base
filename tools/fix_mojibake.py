"""
Replace literal UTF-8 em-dash (U+2014) with LaTeX markup '---' for
maximal portability. Also normalize a few other common Unicode
punctuation that XeLaTeX/pdflatex with inputenc=utf8 may render
inconsistently.
"""

from pathlib import Path

REPLACEMENTS = [
    ('—', '---'),    # em-dash
    ('–', '--'),     # en-dash
    ('‘', "`"),      # left single quote
    ('’', "'"),      # right single quote
    ('“', "``"),    # left double quote
    ('”', "''"),    # right double quote
    ('…', r'\ldots{}'),  # horizontal ellipsis
]

paths = [
    r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper\main.tex",
    r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper_tmlr\main.tex",
]

for p in paths:
    text = Path(p).read_text(encoding='utf-8')
    orig = text
    counts = {}
    for src, dst in REPLACEMENTS:
        c = text.count(src)
        if c:
            counts[src] = c
            text = text.replace(src, dst)
    if text != orig:
        backup = Path(p).with_suffix(Path(p).suffix + '.bak_mojibake')
        if not backup.exists():
            backup.write_text(orig, encoding='utf-8')
        Path(p).write_text(text, encoding='utf-8')
    print(f"{p}: {counts}")
