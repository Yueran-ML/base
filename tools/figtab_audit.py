"""List figures/tables in main body (lines < 1300) with their labels + captions."""
import re
from pathlib import Path

text = Path(r"C:\Users\ASUS\Desktop\文件\学术\7841\base\paper\main.tex").read_text(encoding='utf-8')
lines = text.split('\n')
i = 0
n = 0
while i < len(lines) and i < 1300:
    if lines[i].startswith(r"\begin{figure}") or lines[i].startswith(r"\begin{table}"):
        n += 1
        env = "figure" if "figure" in lines[i] else "table"
        j = i + 1
        cap = lab = None
        while j < len(lines):
            if lines[j].startswith("\\end{" + env + "}"):
                break
            s = lines[j].lstrip()
            if s.startswith("\\caption") and cap is None:
                cap = lines[j].strip()[:100]
            if "\\label{" in lines[j] and lab is None:
                m = re.search(r"\\label\{([^}]+)\}", lines[j])
                if m:
                    lab = m.group(1)
            j += 1
        print(f"#{n} {env:7s} L{i+1:5d}-{j+1:5d}  {lab or '(no label)':30s}  {cap or ''}")
        i = j
    i += 1
