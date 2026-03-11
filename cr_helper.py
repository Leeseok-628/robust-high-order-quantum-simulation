import numpy as np
import scipy.linalg


# ============================================================
# 0. Basic Pauli operators
# ============================================================

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


# ============================================================
# 1. Tensor / Pauli helpers
# ============================================================

def tensor_product(matrices):
    """Tensor product of a list of matrices."""
    out = matrices[0]
    for mat in matrices[1:]:
        out = np.kron(out, mat)
    return out


def pauli_tensor_product(n_qubits, pauli_string):
    """
    Generate a tensor product of Pauli matrices for an n-qubit system.

    Examples
    --------
    pauli_tensor_product(4, "X1")
    pauli_tensor_product(4, "X1Z2Y3")
    """
    pauli_dict = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "I": I,
    }

    ops = [I for _ in range(n_qubits)]

    i = 0
    while i < len(pauli_string):
        p_type = pauli_string[i].upper()
        if p_type not in pauli_dict:
            raise ValueError(f"Invalid Pauli type: {p_type}")

        if i + 1 >= len(pauli_string) or not pauli_string[i + 1].isdigit():
            raise ValueError(
                "Each Pauli must be followed by a qubit index, e.g. 'X1Z2'"
            )

        pos = int(pauli_string[i + 1]) - 1
        if pos < 0:
            raise ValueError("Qubit indices must start at 1.")

        ops[pos] = pauli_dict[p_type]
        i += 2

    return tensor_product(ops)


def construct_hamiltonian_term(term, positions, n_qubits=4):
    """
    Construct nearest-neighbor terms like sum_i term_i term_{i+1}
    over the positions list (excluding the last sentinel, following
    your original implementation).
    """
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in positions[:-1]:
        ops = [I] * n_qubits
        ops[i] = term
        ops[(i + 1) % n_qubits] = term
        H += tensor_product(ops)
    return H


# ============================================================
# 2. Drift / target Hamiltonian builders
# ============================================================

def construct_CR_hamiltonian(n_qubits, J=1.0):
    """
    Construct
        H_0 = J * sum_i X_i Z_{i+1}
    for an open chain of n_qubits.
    """
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = X
        ops[i + 1] = Z
        H += J * tensor_product(ops)

    return H


# ============================================================
# 3. Error metrics
# ============================================================

def gate_error(U, V):
    """
    Phase-aligned unsquared Frobenius error:
        ||U - e^{i phi} V||_F / sqrt(2 d)
    """
    phase = np.angle(np.trace(U @ V.conj().T))
    return np.linalg.norm(U - np.exp(1j * phase) * V, "fro") / np.sqrt(2 * U.shape[0])


def trace_distance(M1, M2, dim=None):
    """
    Trace distance-style quantity from your original helper.
    """
    eigvals = np.linalg.eigvals(M1 - M2)
    return 0.5 * np.sum(np.abs(eigvals))


# ============================================================
# 4. Primitive pulse unitaries
# ============================================================

def full_pulse(H_ctrl, H_0, tp):
    """
    Full pulse:
        exp[-i ( (pi/2) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * ((np.pi / 2) * H_ctrl + tp * H_0))


def reverse_full_pulse(H_ctrl, H_0, tp):
    """
    Reverse full pulse:
        exp[-i ( -(pi/2) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * (-(np.pi / 2) * H_ctrl + tp * H_0))


def half_pulse(H_ctrl, H_0, tp):
    """
    Half pulse:
        exp[-i ( (pi/4) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * ((np.pi / 4) * H_ctrl + tp * H_0))


def reverse_half_pulse(H_ctrl, H_0, tp):
    """
    Reverse half pulse:
        exp[-i ( -(pi/4) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * (-(np.pi / 4) * H_ctrl + tp * H_0))


# --- Backward-compatible aliases ---
def U(H, H_S, tp):
    return full_pulse(H, H_S, tp)


def U_rev(H, H_S, tp):
    return reverse_full_pulse(H, H_S, tp)


def U_T(H, H_S, tp):
    return half_pulse(H, H_S, tp)


def U_T_rev(H, H_S, tp):
    return reverse_half_pulse(H, H_S, tp)


# ============================================================
# 5. Fixed 4-qubit control generators used in CR numerics
# ============================================================

N_CR = 4

Z1234 = sum(pauli_tensor_product(N_CR, f"Z{i}") for i in range(1, 5))
Y1234 = sum(pauli_tensor_product(N_CR, f"Y{i}") for i in range(1, 5))

Z12 = pauli_tensor_product(N_CR, "Z1") + pauli_tensor_product(N_CR, "Z2")
Z23 = pauli_tensor_product(N_CR, "Z2") + pauli_tensor_product(N_CR, "Z3")

Y2 = pauli_tensor_product(N_CR, "Y2")
Y3 = pauli_tensor_product(N_CR, "Y3")

Z24 = pauli_tensor_product(N_CR, "Z2") + pauli_tensor_product(N_CR, "Z4")
Y24 = pauli_tensor_product(N_CR, "Y2") + pauli_tensor_product(N_CR, "Y4")


# ============================================================
# 6. Elementary refocusing sequences
# ============================================================

def R_E_tp(H_0, tp):
    """
    Elementary composite refocusing sequence.
    """
    return half_pulse(Y1234, H_0, tp) @ half_pulse(Z1234, H_0, tp)


def R_E_tp_dag(H_0, tp):
    """
    Hermitian-adjoint companion sequence.
    """
    return reverse_half_pulse(Z1234, H_0, tp) @ reverse_half_pulse(Y1234, H_0, tp)


# ============================================================
# 7. First-order DCGs used in the CR example
# ============================================================

def R_Z1234_DCG(H_0, tp):
    Z12_gate = full_pulse(Z12, H_0, tp)
    Z23_gate = full_pulse(Z23, H_0, tp)
    I_Q = reverse_half_pulse(Z1234, H_0, tp) @ half_pulse(Z1234, H_0, tp)

    return (
        half_pulse(Z1234, H_0, 2 * tp)
        @ Z12_gate @ Z23_gate @ Z12_gate @ Z23_gate @ Z23_gate
        @ I_Q @ Z12_gate @ I_Q @ Z23_gate @ I_Q @ Z12_gate
    )


def R_Y1234_DCG(H_0, tp):
    Y2_gate = full_pulse(Y2, H_0, tp)
    Y3_gate = full_pulse(Y3, H_0, tp)
    I_Q = reverse_half_pulse(Y1234, H_0, tp) @ half_pulse(Y1234, H_0, tp)

    return (
        half_pulse(Y1234, H_0, 2 * tp)
        @ Y2_gate @ Y3_gate @ Y2_gate @ Y3_gate @ Y3_gate
        @ I_Q @ Y2_gate @ I_Q @ Y3_gate @ I_Q @ Y2_gate
    )


def R_Z1234_DCG_neg(H_0, tp):
    """
    Negative-angle variant for the Z1234 block.
    """
    Z12_gate = full_pulse(Z12, H_0, tp)
    Z23_gate = full_pulse(Z23, H_0, tp)
    I_Q = reverse_half_pulse(-Z1234, H_0, tp) @ half_pulse(-Z1234, H_0, tp)

    return (
        half_pulse(-Z1234, H_0, 2 * tp)
        @ Z12_gate @ Z23_gate @ Z12_gate @ Z23_gate @ Z23_gate
        @ I_Q @ Z12_gate @ I_Q @ Z23_gate @ I_Q @ Z12_gate
    )


def R_Y1234_DCG_neg(H_0, tp):
    """
    Negative-angle variant for the Y1234 block.
    """
    Y2_gate = full_pulse(Y2, H_0, tp)
    Y3_gate = full_pulse(Y3, H_0, tp)
    I_Q = reverse_half_pulse(-Y1234, H_0, tp) @ half_pulse(-Y1234, H_0, tp)

    return (
        half_pulse(-Y1234, H_0, 2 * tp)
        @ Y2_gate @ Y3_gate @ Y2_gate @ Y3_gate @ Y3_gate
        @ I_Q @ Y2_gate @ I_Q @ Y3_gate @ I_Q @ Y2_gate
    )


# ============================================================
# 8. Robust composite refocusing blocks
# ============================================================

def R_E_DCG(H_0, tp):
    """
    Full robust Eulerian DCG block.
    """
    return R_Y1234_DCG(H_0, tp) @ R_Z1234_DCG(H_0, tp)


def R_E_DCG_dag(H_0, tp):
    """
    Hermitian-adjoint robust Eulerian DCG block.
    """
    return R_Z1234_DCG_neg(H_0, tp) @ R_Y1234_DCG_neg(H_0, tp)


# ============================================================
# 9. Two-subspace variants used in CR numerics
# ============================================================

def R_Z24_DCG(H_0, tp):
    Z12_gate = full_pulse(Z12, H_0, tp)
    Z23_gate = full_pulse(Z23, H_0, tp)
    I_Q = reverse_full_pulse(Z24, H_0, tp) @ full_pulse(Z24, H_0, tp)

    return (
        full_pulse(Z24, H_0, 2 * tp)
        @ Z12_gate @ Z23_gate @ Z12_gate @ Z23_gate @ Z23_gate
        @ I_Q @ Z12_gate @ I_Q @ Z23_gate @ I_Q @ Z12_gate
    )


def R_Y24_DCG(H_0, tp):
    Y2_gate = full_pulse(Y2, H_0, tp)
    Y3_gate = full_pulse(Y3, H_0, tp)
    I_Q = reverse_half_pulse(Y24, H_0, tp) @ half_pulse(Y24, H_0, tp)

    return (
        half_pulse(Y24, H_0, 2 * tp)
        @ Y2_gate @ Y3_gate @ Y2_gate @ Y3_gate @ Y3_gate
        @ I_Q @ Y2_gate @ I_Q @ Y3_gate @ I_Q @ Y2_gate
    )


# ============================================================
# 10. General DCG wrapper
# ============================================================

def U_DCG(H_target, H_0, tp):
    """
    General DCG for an arbitrary target control Hamiltonian.
    """
    Z12_gate = full_pulse(Z12, H_0, tp)
    Z23_gate = full_pulse(Z23, H_0, tp)
    I_Q = reverse_full_pulse(H_target, H_0, tp) @ full_pulse(H_target, H_0, tp)

    return (
        full_pulse(H_target, H_0, 2 * tp)
        @ Z12_gate @ Z23_gate @ Z12_gate @ Z23_gate @ Z23_gate
        @ I_Q @ Z12_gate @ I_Q @ Z23_gate @ I_Q @ Z12_gate
    )


# ============================================================
# 11. Small utility for log-log slope extraction
# ============================================================

def slope_loglog(x, y, floor=1e-18):
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), floor)
    return np.polyfit(np.log10(x), np.log10(y), 1)[0]