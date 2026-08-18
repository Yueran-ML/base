import re
text = open('paper/main.tex', encoding='utf-8').read()
lines = text.splitlines()
pat = re.compile(r'[A-Za-z\)\}]---[A-Za-z\\]')
suspect = [(i+1, l.strip()[:140]) for i, l in enumerate(lines) if pat.search(l)]
print(f'em-dash prose lines: {len(suspect)}')
for ln, l in suspect:
    print(ln, l)
