import re
text = open('paper/main.tex', encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = set(re.findall(r'\\(?:ref|autoref|Cref|cref|eqref)\{([^}]+)\}', text))
print('labels:', len(labels), 'refs:', len(refs))
print('missing labels:', refs - labels)
cites = set(re.findall(r'\\cite[a-z]*\{([^}]+)\}', text))
all_cites = set()
for c in cites:
    for k in c.split(','):
        all_cites.add(k.strip())
bib = open('paper/refs.bib', encoding='utf-8').read()
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
print('cites:', len(all_cites), 'bibkeys:', len(bibkeys))
print('missing bibkeys:', all_cites - bibkeys)
