"""
Convert narrative cites like 'Author et al.~\cite{key}' or 'Author and Author~\cite{key}'
into \citet{key}, while leaving parenthetical \cite{key} alone (those will be
mass-converted to \citep{key} separately).

Run on a single .tex file in place. Backup is written next to the file.
"""

import re
import sys
import shutil
from pathlib import Path


NARRATIVE_RE = re.compile(
    r'([A-Z][a-zA-Z\']+(?: et al\.| and [A-Z][a-zA-Z]+)?)~\\cite\{([^}]+)\}'
)


def fix_file(path: str):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    backup = p.with_suffix(p.suffix + '.bak_cites')
    if not backup.exists():
        shutil.copy(p, backup)

    matches = NARRATIVE_RE.findall(text)
    print(f'{path}: found {len(matches)} narrative-cite patterns')

    new_text = NARRATIVE_RE.sub(lambda m: '\\citet{' + m.group(2) + '}', text)

    # Now mass-convert remaining bare \cite{...} -> \citep{...} EXCEPT those
    # we already turned into \citet (those have \citet which won't match \cite{).
    bare_count = len(re.findall(r'\\cite\{', new_text))
    print(f'  remaining bare \\cite{{}}: {bare_count} -> will become \\citep{{}}')
    new_text = re.sub(r'\\cite\{', r'\\citep{', new_text)

    p.write_text(new_text, encoding='utf-8')
    print(f'  wrote: {path}')


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        fix_file(arg)
