import numpy as np
import scipy.linalg as la

# ------------------------------------------------
# --- Basic 2-site Pauli pair constructors ---
# ------------------------------------------------
def XX_pair(site, i, j):
    """Return XX term acting on (i,j) in an n-site chain."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I]*site
    ops[i] = X
    ops[j] = X
    return kron_all(ops)

def ZZ_pair(site, i, j):
    """Return ZZ term acting on (i,j) in an n-site chain."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I]*site
    ops[i] = Z
    ops[j] = Z
    return kron_all(ops)

def YY_pair(site, i, j):
    """Return YY term acting on (i,j) in an n-site chain."""
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = [I]*site
    ops[i] = Y
    ops[j] = Y
    return kron_all(ops)

# ------------------------------------------------
# --- Utility: tensor (Kronecker) product over list ---
# ------------------------------------------------
def kron_all(op_list):
    """Efficient Kronecker product over list of matrices."""
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return np.matrix(result)


def gate_fidelity(U,V,site):
    return 1-abs(np.trace(U@V.conj().T))/2**site

def trace_distance(M1,M2,dim):

    eig,vec = np.linalg.eig(M1 -M2)

    trace = 0
    for i in range(dim):
        trace = trace + abs(eig[i])

    return trace/2