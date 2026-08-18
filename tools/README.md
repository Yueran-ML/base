# tools/

Reusable Python utilities for paper hygiene. All read-only or
in-place edit (with explicit backups to `*.bak_*`).

## Audit utilities

- **`format_audit.py`** — counts `\cite/\citet/\citep`, checks for
  duplicate-rendering narrative-cite patterns, surfaces
  Ethics/Reproducibility/LLM/Broader-Impact statement presence,
  flags work-log phrases ("has been executed"), reports section
  line numbers.
- **`xref_audit.py`** — lists all `\label{...}` / `\ref{...}` pairs
  across both papers; flags dangling refs and unused labels.
- **`figtab_audit.py`** — enumerates main-body figures/tables (lines
  &lt; 1300) with their labels and captions; useful for tracking
  which floats live in main vs. appendix.

## Edit utilities

- **`fix_citations.py`** — converts `Author et al.~\cite{key}` to
  `\citet{key}` and bare `\cite{key}` to `\citep{key}`. Writes a
  `*.bak_cites` backup. Idempotent on already-converted files.
- **`fix_mojibake.py`** — replaces literal Unicode punctuation
  (em-dash, en-dash, smart quotes, ellipsis) with portable LaTeX
  markup. Writes a `*.bak_mojibake` backup.

## Usage

All scripts run with the project venv:

```
.venv/Scripts/python.exe tools/format_audit.py
.venv/Scripts/python.exe tools/xref_audit.py
.venv/Scripts/python.exe tools/figtab_audit.py
.venv/Scripts/python.exe tools/fix_citations.py paper/main.tex
.venv/Scripts/python.exe tools/fix_mojibake.py
```

The audit scripts hard-code paths to `paper/main.tex` and
`paper_tmlr/main.tex`. The fix scripts take file paths as arguments
(or hard-code both files for the mojibake one).
