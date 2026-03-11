# Robust high-order quantum simulation using finite-width pulses

This repository contains the numerical simulation code accompanying our paper on higher-order pulse-sequence constructions for Hamiltonian simulation.

## Files

- `construction1_ising.py` — Construction 1, Ising example 
- `construction1_cr.py` — Construction 1, CR-to-Heisenberg example 
- `construction2_hb.py` — Construction 2 / anisotropic Heisenberg example 
- `construction1_cr_mpf.py` — MPF observable benchmark for the CR example
- `construction2_hb_mpf.py` — MPF observable benchmark for the anisotropic Heisenberg example (Fig 10 in the paper)

- `Eulerian.py` — Eulerian-cycle generation and related pulse-sequence routines
- `cr_helper.py` — Helper functions for `construction1_cr_mpf.py`

- `construction1_ising.ipynb` — Construction 1, Ising example (reproduces Fig. 3 in the paper)
- `construction1_cr.ipynb` — Construction 1, CR-to-Heisenberg example  (reproduces Figs. 4 and 9 in the paper)
- `construction2_hb.ipynb` — Construction 2 / anisotropic Heisenberg example (reproduces Figs. 5 and 10 in the paper)

## Dependencies

pip install numpy scipy matplotlib
