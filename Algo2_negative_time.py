import numpy as np
import scipy
import qutip
import matplotlib.pyplot as plt
import math
from Eulerian import *
from Algo2_helper import *

# --- System size ---
site = 4

# --- Pauli matrices ---
X = np.matrix([[0, 1], [1, 0]], dtype=complex)
Y = np.matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = np.matrix([[1, 0], [0, -1]], dtype=complex)
I = np.matrix(np.eye(2, dtype=complex))

# --- Construct time-independent Hamiltonian H0 ---
HXX = HYY = HZZ = 0
for i in range(site - 1):
    HXX += XX_pair(site, i, i + 1)
    HZZ += ZZ_pair(site, i, i + 1)
    HYY += -XX_pair(site, i, i + 1).dot(ZZ_pair(site, i, i + 1))
H0 = HXX + HYY + HZZ

# ================================================================
#  Refocusing Sequence Construction (Eulerian + XY8 Symmetrized)
# ================================================================

# --- Base XY8 pattern for local refocusing ---
XY8 = [
    ('I', 'X', 'I', 'X'), ('I', 'Y', 'I', 'Y'),
    ('I', 'X', 'I', 'X'), ('I', 'Y', 'I', 'Y'),
    ('I', 'Y', 'I', 'Y'), ('I', 'X', 'I', 'X'),
    ('I', 'Y', 'I', 'Y'), ('I', 'X', 'I', 'X'),
]
XY8_CYCLE = [create_sum_matrix(pauli) for pauli in XY8]

# --- Eulerian cycle on 4-qubit Pauli group ---
cycle = find_eulerian_cycle(generate_cayley_graph(4))
Euler_cycle = compute_transition_paulis(cycle)
Euler_cycle_mat = [create_sum_matrix(pauli) for pauli in Euler_cycle]


# --- Finite-width pulses ---
def U(H, H0, tp):
    """Forward finite-width π/2 pulse under drift H0."""
    return scipy.linalg.expm(-1j * (np.pi / 2 * H + tp * H0))

def U_rev(H, H0, tp):
    """Reverse finite-width −π/2 pulse under drift H0."""
    return scipy.linalg.expm(-1j * (-np.pi / 2 * H + tp * H0))


# --- XY8-symmetrized DD block ---
def U_XY8_symm(H0, tp, tau):
    """Symmetrized XY8 cycle with finite-width pulses."""
    U_tot = np.eye(2**4, dtype=complex)
    ftau = scipy.linalg.expm(-1j * H0 * tau)

    # Forward pass
    for M in XY8_CYCLE:
        U_tot = (U(M, H0, tp) @ ftau) @ U_tot

    # Reverse pass (time-symmetric)
    for M in reversed(XY8_CYCLE):
        U_tot = (ftau @ U_rev(M, H0, tp)) @ U_tot

    return U_tot


# --- Eulerian-symmetrized concatenation (k=2) ---
def refocus_4(tau, tp):
    """
    Implements negative-time evotluion via concatenation of symmetrized XY8 (Eulerian) blocks.
    """
    U_ref = np.eye(2**4, dtype=complex)
    XY8_symm_block = U_XY8_symm(H0, tp, tau)

    # Forward Eulerian traversal
    for M in Euler_cycle_mat:
        U_ref = (U(M, H0, tp) @ XY8_symm_block) @ U_ref

    # Reverse traversal (time-symmetrized)
    for M in reversed(Euler_cycle_mat):
        U_ref = (XY8_symm_block @ U_rev(M, H0, tp)) @ U_ref

    return U_ref @ scipy.linalg.expm(1j * H0 * tau)


#If needed (e.g., for very small t_p):

def project_to_unitary(A):
    """project to the closest unitary A (polar decomposition)."""
    U, _ = scipy.linalg.polar(A)
    return U


# ------------------------------------------------
# Eulerian Symmetrized CDD (ESCDD) for k = 2
# with periodic re-unitarization (fixed parameters)
# ------------------------------------------------
def refocus_4_project(tau, tp):
    """
    tau, tp: durations
    Project to the unitary via polar decomposition periodically.
    This is used to handle the numerical instability due to numerous matrix multiplications.
    """
    assert tau > 0, "tau must be positive"

    n = 2**4
    ESCDD1_symm = np.eye(n, dtype=complex)
    XY8_symm_block = U_XY8_symm(H0, tp, tau)

    reunitarize_every = 256   # frequency of projection
    mults = 0

    # Forward pass
    for M in Euler_cycle_mat:
        ESCDD1_symm = (U(M, H0, tp) @ XY8_symm_block) @ ESCDD1_symm
        mults += 2
        if mults % reunitarize_every == 0:
            ESCDD1_symm = project_to_unitary(ESCDD1_symm)

    # Reverse pass
    for M in reversed(Euler_cycle_mat):
        ESCDD1_symm = (XY8_symm_block @ U_rev(M, H0, tp)) @ ESCDD1_symm
        mults += 2
        if mults % reunitarize_every == 0:
            ESCDD1_symm = project_to_unitary(ESCDD1_symm)

    ESCDD1_symm = ESCDD1_symm @ scipy.linalg.expm(1j * H0 * tau)
    #ESCDD1_symm = project_to_unitary(ESCDD1_symm)
    return ESCDD1_symm
