import numpy as np
import scipy.linalg

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def construct_CR_hamiltonian(N, J=1.0):
    """
    Construct the H_S = J \sum_i X_i Z_{i+1} Hamiltonian for a chain of N qubits.

    Parameters:
    - N: number of qubits
    - J: coupling strength (default=1.0)

    Returns:
    - H: (2^N x 2^N) complex NumPy array representing the Hamiltonian
    """
    # Define Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # Initialize Hamiltonian as zero matrix
    H = np.zeros((2**N, 2**N), dtype=complex)

    # Build H = J * sum_i X_i Z_{i+1}
    for i in range(N - 1):
        # construct tensor product for X_i Z_{i+1}
        ops = [I] * N
        ops[i] = X
        ops[i + 1] = Z
        # tensor product chain
        H_term = ops[0]
        for op in ops[1:]:
            H_term = np.kron(H_term, op)
        H += J * H_term

    return H

def construct_hamiltonian_term(term, positions, n_qubits=4):
    """Construct terms like X_i X_{i+1}."""
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in positions[:-1]:
        ops = [I] * n_qubits
        ops[i] = term
        ops[(i + 1) % n_qubits] = term  # Periodic boundary
        H += tensor_product(ops)
    return H


def tensor_product(matrices):
    """Tensor product of matrices."""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def pauli_tensor_product(N, pauli_string):
    """
    Generate a tensor product of Pauli matrices for an N-qubit system (NumPy version).

    Parameters:
    - N: Total number of qubits
    - pauli_string: String like "X1X2", "Y3", "X1Z2Y3", etc.

    Returns:
    - H: (2^N x 2^N) NumPy array
    """
    # Define Pauli matrices and identity
    pauli_dict = {
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'I': np.eye(2, dtype=complex)
    }

    # Start with all identities
    ops = [pauli_dict['I'] for _ in range(N)]

    # Parse input string (e.g., "X1Z2Y3")
    i = 0
    while i < len(pauli_string):
        p_type = pauli_string[i].upper()
        if p_type not in pauli_dict:
            raise ValueError(f"Invalid Pauli type: {p_type}")

        if i + 1 >= len(pauli_string) or not pauli_string[i + 1].isdigit():
            raise ValueError("Each Pauli must be followed by a qubit index, e.g. 'X1Z2'")

        pos = int(pauli_string[i + 1]) - 1  # convert to 0-based index
        if pos < 0 or pos >= N:
            raise ValueError(f"Qubit index {pos+1} out of range for {N}-qubit system")

        ops[pos] = pauli_dict[p_type]
        i += 2  # move to next Pauli specifier

    # Compute tensor product
    H = ops[0]
    for op in ops[1:]:
        H = np.kron(H, op)

    return H

def gate_fidelity(U,V,site):
    return 1-abs(np.trace(U@V.conj().T))/2**site

def trace_distance(M1,M2,dim):

    eig,vec = np.linalg.eig(M1 -M2)

    trace = 0
    for i in range(dim):
        trace = trace + abs(eig[i])

    return trace/2


# ================================================================
# --- Basic Unitary building blocks ---
# ================================================================

def U(H, H_S, tp):
    """Full pulse: +π/2 rotation on H with drift H_S for duration tp."""
    return scipy.linalg.expm(-1j * (np.pi/2 * H + tp * H_S))

def U_rev(H, H_S, tp):
    """Reverse pulse: -π/2 rotation on H."""
    return scipy.linalg.expm(-1j * (-np.pi/2 * H + tp * H_S))

def U_T(H, H_S, tp):
    """Half pulse: +π/4 rotation on H."""
    return scipy.linalg.expm(-1j * (np.pi/4 * H + tp * H_S))

def U_T_rev(H, H_S, tp):
    """Half reverse pulse: -π/4 rotation on H."""
    return scipy.linalg.expm(-1j * (-np.pi/4 * H + tp * H_S))


# ================================================================
# --- Multi-qubit Pauli operators ---
# ================================================================

Z1234 = sum(pauli_tensor_product(4, f"Z{i}") for i in range(1, 5))
Y1234 = sum(pauli_tensor_product(4, f"Y{i}") for i in range(1, 5))

Z12 = pauli_tensor_product(4, "Z1") + pauli_tensor_product(4, "Z2")
Z23 = pauli_tensor_product(4, "Z2") + pauli_tensor_product(4, "Z3")

Y2, Y3 = pauli_tensor_product(4, "Y2"), pauli_tensor_product(4, "Y3")

Z24 = pauli_tensor_product(4, "Z2") + pauli_tensor_product(4, "Z4")
Y24 = pauli_tensor_product(4, "Y2") + pauli_tensor_product(4, "Y4")


# ================================================================
# --- Elementary Refocusing Sequences ---
# ================================================================

def R_E_tp(H_S, tp):
    """Elementary composite refocusing (Eulerian-like) using Y and Z blocks."""
    return U_T(Y1234, H_S, tp) @ U_T(Z1234, H_S, tp)

def R_E_tp_dag(H_S, tp):
    """Hermitian adjoint of R_E_tp."""
    return U_T_rev(Z1234, H_S, tp) @ U_T_rev(Y1234, H_S, tp)


# ================================================================
# --- DCG Constructions for various target Hamiltonians ---
# ================================================================

def R_Z1234_DCG(H_S, tp):
    Z12gate, Z23gate = U(Z12, H_S, tp), U(Z23, H_S, tp)
    I_Q = U_T_rev(Z1234, H_S, tp) @ U_T(Z1234, H_S, tp)
    return (
        U_T(Z1234, H_S, 2*tp)
        @ Z12gate @ Z23gate @ Z12gate @ Z23gate @ Z23gate
        @ I_Q @ Z12gate @ I_Q @ Z23gate @ I_Q @ Z12gate
    )

def R_Y1234_DCG(H_S, tp):
    Y2gate, Y3gate = U(Y2, H_S, tp), U(Y3, H_S, tp)
    I_Q = U_T_rev(Y1234, H_S, tp) @ U_T(Y1234, H_S, tp)
    return (
        U_T(Y1234, H_S, 2*tp)
        @ Y2gate @ Y3gate @ Y2gate @ Y3gate @ Y3gate
        @ I_Q @ Y2gate @ I_Q @ Y3gate @ I_Q @ Y2gate
    )


# --- Negative-angle variants (for reversed rotation) ---

def R_Z1234_DCG_neg(H_S, tp):
    Z12gate, Z23gate = U(Z12, H_S, tp), U(Z23, H_S, tp)
    I_Q = U_T_rev(-Z1234, H_S, tp) @ U_T(-Z1234, H_S, tp)
    return (
        U_T(-Z1234, H_S, 2*tp)
        @ Z12gate @ Z23gate @ Z12gate @ Z23gate @ Z23gate
        @ I_Q @ Z12gate @ I_Q @ Z23gate @ I_Q @ Z12gate
    )

def R_Y1234_DCG_neg(H_S, tp):
    Y2gate, Y3gate = U(Y2, H_S, tp), U(Y3, H_S, tp)
    I_Q = U_T_rev(-Y1234, H_S, tp) @ U_T(-Y1234, H_S, tp)
    return (
        U_T(-Y1234, H_S, 2*tp)
        @ Y2gate @ Y3gate @ Y2gate @ Y3gate @ Y3gate
        @ I_Q @ Y2gate @ I_Q @ Y3gate @ I_Q @ Y2gate
    )


# ================================================================
# --- DCG Robust Sequences ---
# ================================================================

def R_E_DCG(H_S, tp):
    """Full robust Eulerian DCG."""
    return R_Y1234_DCG(H_S, tp) @ R_Z1234_DCG(H_S, tp)

def R_E_DCG_dag(H_S, tp):
    """Hermitian adjoint of robust Eulerian DCG."""
    return R_Z1234_DCG_neg(H_S, tp) @ R_Y1234_DCG_neg(H_S, tp)


# ================================================================
# --- Two-qubit subspace DCG variants (e.g., for 2-4 coupling) ---
# ================================================================

def R_Z24_DCG(H_S, tp):
    Z12gate, Z23gate = U(Z12, H_S, tp), U(Z23, H_S, tp)
    I_Q = U_rev(Z24, H_S, tp) @ U(Z24, H_S, tp)
    return (
        U(Z24, H_S, 2*tp)
        @ Z12gate @ Z23gate @ Z12gate @ Z23gate @ Z23gate
        @ I_Q @ Z12gate @ I_Q @ Z23gate @ I_Q @ Z12gate
    )

def R_Y24_DCG(H_S, tp):
    Y2gate, Y3gate = U(Y2, H_S, tp), U(Y3, H_S, tp)
    I_Q = U_T_rev(Y24, H_S, tp) @ U_T(Y24, H_S, tp)
    return (
        U_T(Y24, H_S, 2*tp)
        @ Y2gate @ Y3gate @ Y2gate @ Y3gate @ Y3gate
        @ I_Q @ Y2gate @ I_Q @ Y3gate @ I_Q @ Y2gate
    )


# ================================================================
# --- General DCG (arbitrary H) ---
# ================================================================

def U_DCG(H, H_S, tp):
    """General DCG for arbitrary H (4-qubit example)."""
    Z12gate, Z23gate = U(Z12, H_S, tp), U(Z23, H_S, tp)
    I_Q = U_rev(H, H_S, tp) @ U(H, H_S, tp)
    return (
        U(H, H_S, 2*tp)
        @ Z12gate @ Z23gate @ Z12gate @ Z23gate @ Z23gate
        @ I_Q @ Z12gate @ I_Q @ Z23gate @ I_Q @ Z12gate
    )