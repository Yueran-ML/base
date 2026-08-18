import re
text = open('E:/main.tex', encoding='utf-8').read()
lines = text.splitlines()
pat = re.compile(r'[A-Za-z\)\}]---[A-Za-z\\]')
suspect = [(i+1, l.strip()[:120]) for i, l in enumerate(lines) if pat.search(l)]
print(f'em-dash prose lines in E:/main.tex: {len(suspect)}')
for ln, l in suspect[:10]:
    print(ln, l)
