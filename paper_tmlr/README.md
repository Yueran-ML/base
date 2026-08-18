# TMLR submission package

This directory mirrors `../paper/` but with a TMLR-formatted preamble and
title block.

## What you must add before compiling

The TMLR style files are not included here (they belong to the JmlrOrg
template repository). Download them from:

  https://github.com/JmlrOrg/tmlr-style-file

and place the following files alongside `main.tex`:

  * `tmlr.sty`     -- TMLR document class style (page layout, title block)
  * `tmlr.bst`     -- TMLR bibliography style for natbib
  * `fancyhdr.sty` -- usually already in TeX Live, ship a copy if missing

Or, on most TeX Live installations, run:

    kpsewhich tmlr.sty
    kpsewhich tmlr.bst

If both return paths, you do not need to copy them locally.

## Switching between submission modes

The default `\usepackage{tmlr}` renders the **anonymous submission**
format used during peer review.

For the **camera-ready (accepted)** version, change the line in
`main.tex` from:

    \usepackage{tmlr}

to:

    \usepackage[accepted]{tmlr}

For a **non-anonymous preprint**, use:

    \usepackage[preprint]{tmlr}

## Author block

The current author block is the anonymous placeholder:

    \author{%
      \name Anonymous Author \\
      \addr Anonymous Affiliation \\
      \email anon@example.com
    }

Replace with the real author / affiliation / email when going to
camera-ready or preprint.

## Compile

Standard sequence:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Or with latexmk:

    latexmk -pdf main

## Differences from the ICLR-style version in `../paper/`

  * Replaced `\usepackage[margin=1in]{geometry}` + manual fontenc/inputenc
    with `\usepackage{tmlr}` (which sets the standard TMLR layout).
  * Removed the `\textbf{...}` wrap around the title (TMLR formats the
    title font itself).
  * Replaced `\bibliographystyle{IEEEtran}` with `\bibliographystyle{tmlr}`.
  * Added the `\editor{Anonymous}` placeholder TMLR expects.
  * Removed `\usepackage{times}` (TMLR sets a different default font).
  * Removed the redundant `\usepackage{hyperref}` and `\usepackage{url}`
    lines (loaded by `tmlr.sty`).

All body text, figures, tables, and `refs.bib` entries are unchanged.

## Differences from `paper_upload.zip`

`paper_upload.zip` (in the project root) is the ICLR-style anonymous
submission format we have been iterating on.
`paper_upload_tmlr.zip` is the TMLR-formatted variant of the same content.
The two zips share `figures/` and `refs.bib` byte-for-byte; only the
preamble of `main.tex` differs.
