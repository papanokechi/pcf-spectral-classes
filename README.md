# PCF Spectral Classes — Supplementary Data and Code

Supplementary material for:

> *Spectral Classes of Polynomial Continued Fractions:*
> *A Hypergeometric Framework and Arithmetic Obstructions*  
> Submitted to Experimental Mathematics, April 2026.

## Contents

| File | Description |
|------|-------------|
| `spectral_class_phase1.py` | Computes the Phase 1 taxonomy: spectral class tuple `(d, Λ, Δ, ρ, τ)` for all 37 families |
| `spectral_class_phase2_congruence.py` | Computes `z0`-orbit analysis, `PGL2(Q)` orbit assignments, and obstruction diagnostics |
| `pcf_spectral_phase1_taxonomy.csv` | Complete 37-family taxonomy with numerical limits to 50+ digits |
| `pcf_spectral_phase2_congruence.csv` | Hypergeometric `z0` data and `2F1` fit results for all identified `d=1` families |

## Requirements

Python 3.10 or later.

Install dependencies:

```bash
pip install mpmath sympy
```

Tested with `mpmath 1.3`, `sympy 1.12`.

## Reproducing the results

Reproduce Table 1 (37-family taxonomy):

```bash
python spectral_class_phase1.py
```

Expected runtime: approximately 5 minutes.  
Output: `pcf_spectral_phase1_taxonomy.csv`

Reproduce `z0` analysis and obstruction diagnostics:

```bash
python spectral_class_phase2_congruence.py
```

Expected runtime: approximately 10 minutes.  
Output: `pcf_spectral_phase2_congruence.csv`

## Key results reproduced by these scripts

- Poincaré characteristic lemma: verified to 65-digit precision for all `d=1` families (`residual < 1e-64` in all cases)
- Catalan logarithmic obstruction: `3/(2n)` Frobenius residual, PSLQ exclusion at 220-digit precision
- Hypergeometric `z0` framework: `15/15` identified `d=1` families admit a `2F1` representation
- Three-tier obstruction hierarchy: Catalan (`Delta=0`), `V_quad` (`d=2`, confluent-Heun), Apéry-`zeta(3)` (`d=3`, Picard-Fuchs)

## Citation

If you use this code or data, please cite:  
`[Citation will be updated upon acceptance]`
