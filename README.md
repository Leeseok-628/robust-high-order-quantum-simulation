# Robust high-order quantum simulation using finite-width pulses
Building on the close connection between pulse sequences and Trotter formulas, this repository provides a robust high-order framework for simulating or engineering target dynamics using finite-width pulse sequences. The framework takes as input a base control sequence that simulates a target Hamiltonian to first order in its total evolution time, and systematically upgrades it to a longer sequence that achieves higher-order accuracy in the same total time.  In general, the framework supports two settings:  
(1) a base pulse sequence with ideal instantaneous pulses (*Algorithm 1*), and  
(2) a base robust finite-width pulse sequence (*Algorithm 2*).  
In addition, multi-product formulas can be utilized to achieve similar high-order scaling without the need to implement negative-time evolutions.  All codes for the three numerical simulations presented in the paper are provided in this repository.


## Files

### Helper Python Files
- **`Eulerian.py`**  
  Constructs a Cayley graph from a finite group and its generators, and finds an Eulerian cycle.  
  This cycle is used to build robust pulse sequences for implementing negative-time evolutions.  
  (More general robust sequences can also be constructed using this framework.)

- **`CR_helper.py`**  
  Helper functions for the notebook **`Algorithm 1 CR to Heisenberg.ipynb`**.

- **`Algo2_helper.py`**, **`Algo2_negative_time.py`**  
  Supporting modules for the notebook **`Algorithm 2 Heisenberg Model.ipynb`**.

---

### Jupyter Notebooks

Notebooks for reproducing the figures and numerical results presented in the paper:

- **`Algorithm 1 Ising Model.ipynb`**  
  Demonstrates Algorithm 1 by simulating an inhomogeneous Ising chain from a homogeneous one using local X controls.

- **`Algorithm 1 CR to Heisenberg.ipynb`**  
  Demonstrates Algorithm 1 for simulating a Heisenberg chain from a cross-resonance (CR) Hamiltonian with local controls.

- **`Algorithm 2 Heisenberg Model.ipynb`**  
  Demonstrates Algorithm 2 by simulating an inhomogeneous Heisenberg chain from a homogeneous one.
