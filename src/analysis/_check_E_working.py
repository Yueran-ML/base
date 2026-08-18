import re
text = open('paper/main_E_working.tex', encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = set(re.findall(r'\\(?:ref|autoref|Cref|cref|eqref)\{([^}]+)\}', text))
print('labels:', len(labels), 'refs:', len(refs))
print('missing labels:', refs - labels)
cites = set(re.findall(r'\\cite[a-z]*\{([^}]+)\}', text))
all_cites = set()
for c in cites:
    for k in c.split(','):
        all_cites.add(k.strip())
bib_path = 'paper/refs.bib'
import os
if os.path.exists(bib_path):
    bib = open(bib_path, encoding='utf-8').read()
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
    print('cites:', len(all_cites), 'bibkeys:', len(bibkeys))
    print('missing bibkeys:', all_cites - bibkeys)
