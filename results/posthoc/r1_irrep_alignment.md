# R1/R9 - Top-K irrep-mean vs max-harmonic Fourier alignment

For Z/p the non-trivial irreps are 1-D characters; as real subspaces these are the 2D Fourier planes already used by F_raw.
The unrestricted mean over all (p-1)/2 irreps is degenerate after centering: by Parseval the variance fractions sum to 1, so the mean is the constant 2/(p-1) and carries no information. 
We therefore contrast the canonical max-harmonic scoring (top-1) against the average of the three most-aligned irrep fractions (top-3 mean).

Cells analysed: 8; tau_F agreement within 500 steps: 2/8.

|delta| (top-3 - max-harmonic) summary across 8 cells:
- median = 2000 steps
- max    = 6000 steps

Interpretation:
- If top-1 and top-3 mean agree to within the 500-step measurement bin for the bulk of cells, the choice of single-vs- few-harmonic scoring is not load-bearing for tau_F and the G<F ordering survives within the cyclic-group irrep family.
- A systematic offset (e.g. top-3 consistently later) would indicate that the canonical max-harmonic detector locks onto early dominant-harmonic structure that the broader irrep average does not yet register.
