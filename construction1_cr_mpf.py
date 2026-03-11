import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from construction1_cr import Tro1, Tro2, H_targ, N_QUBITS


# ================================================================
# Basic single-qubit operators
# ================================================================

I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

PAULI_MAP = {
    "I": I2,
    "X": X2,
    "Y": Y2,
    "Z": Z2,
}


# ================================================================
# Basic helpers
# ================================================================

def expval(O, rho):
    """
    Expectation value Tr(O rho), returned as a real scalar when possible.
    """
    val = np.trace(O @ rho)
    return np.real_if_close(val).item()


def rho_from_ket(ket):
    """
    Density matrix from a state vector.
    """
    ket = np.asarray(ket, dtype=complex).reshape(-1)
    return np.outer(ket, ket.conj())


def computational_basis_ket(bitstring):
    """
    Build a computational-basis product state from a bitstring.

    Example
    -------
    computational_basis_ket("0101")
    """
    if len(bitstring) != N_QUBITS:
        raise ValueError(f"bitstring must have length {N_QUBITS}")

    mapping = {"0": ket0, "1": ket1}

    try:
        state = mapping[bitstring[0]]
        for b in bitstring[1:]:
            state = np.kron(state, mapping[b])
    except KeyError:
        raise ValueError("bitstring must contain only '0' and '1'")

    return state


def local_observable(pauli_map, n_qubits=N_QUBITS):
    """
    Build a tensor-product observable from a dictionary.

    Parameters
    ----------
    pauli_map : dict
        Maps qubit index -> local 2x2 operator or "I"/"X"/"Y"/"Z".
        Example: {0: "X", 1: "X"} gives X_1 X_2 in paper notation.
    """
    ops = []
    for q in range(n_qubits):
        op = pauli_map.get(q, I2)
        if isinstance(op, str):
            if op not in PAULI_MAP:
                raise ValueError(f"Unsupported Pauli label '{op}'")
            op = PAULI_MAP[op]
        ops.append(op)

    O = ops[0]
    for op in ops[1:]:
        O = np.kron(O, op)
    return O


def observable_from_pauli_string(pauli_string):
    """
    Build an observable from a Pauli string such as:
        "XXII", "ZZII", "XIZY"

    Length must equal N_QUBITS.
    """
    pauli_string = pauli_string.upper().replace(" ", "")
    if len(pauli_string) != N_QUBITS:
        raise ValueError(
            f"Pauli string must have length {N_QUBITS}, got {len(pauli_string)}"
        )

    try:
        ops = [PAULI_MAP[ch] for ch in pauli_string]
    except KeyError:
        raise ValueError("Pauli string must contain only I, X, Y, Z")

    O = ops[0]
    for op in ops[1:]:
        O = np.kron(O, op)
    return O


def resolve_observable(observable=None):
    """
    Flexible observable resolver.

    Allowed inputs
    --------------
    - None:
        defaults to "XXII"
    - str:
        Pauli string, e.g. "XXII", "ZZII", "XIZY"
    - dict:
        e.g. {0: "X", 1: "X"} or {0: X2, 1: X2}
    - ndarray:
        full observable matrix
    """
    if observable is None:
        return observable_from_pauli_string("XXII")

    if isinstance(observable, str):
        return observable_from_pauli_string(observable)

    if isinstance(observable, dict):
        return local_observable(observable)

    O = np.asarray(observable, dtype=complex)
    dim = 2 ** N_QUBITS
    if O.shape != (dim, dim):
        raise ValueError(f"Observable must have shape {(dim, dim)}")
    return O


def resolve_initial_state(initial_state=None):
    """
    Flexible initial-state resolver.

    Allowed inputs
    --------------
    - None:
        defaults to bitstring "0101"
    - str:
        computational basis bitstring, e.g. "0101"
    - 1D ndarray:
        treated as a state vector |psi>
    - 2D ndarray:
        treated as a density matrix rho
    """
    dim = 2 ** N_QUBITS

    if initial_state is None:
        return rho_from_ket(computational_basis_ket("0101"))

    if isinstance(initial_state, str):
        return rho_from_ket(computational_basis_ket(initial_state))

    arr = np.asarray(initial_state, dtype=complex)

    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"State vector must have length {dim}")
        return rho_from_ket(arr)

    if arr.ndim == 2:
        if arr.shape != (dim, dim):
            raise ValueError(f"Density matrix must have shape {(dim, dim)}")
        return arr

    raise ValueError(
        "initial_state must be None, a bitstring, a state vector, or a density matrix"
    )


def target_unitary(T):
    """
    Exact target evolution under the Heisenberg Hamiltonian.
    """
    return scipy.linalg.expm(-1j * H_targ * T)


# ================================================================
# MPF expectation-value estimator
# ================================================================

def mpf_expectation(O, rho0, klist, clist, T, tp):
    """
    Returns the MPF estimator

        sum_j c_j Tr[ O rho_j(T) ],

    where
        rho_j(T) = U_j rho0 U_j^\dagger,
        U_j = [Tro2(T / k_j, tp, "DCG")]^{k_j}.
    """
    if len(klist) != len(clist):
        raise ValueError("klist and clist must have the same length")

    est = 0.0
    for k_j, c_j in zip(klist, clist):
        if int(k_j) != k_j or k_j <= 0:
            raise ValueError("All k_j must be positive integers")

        U_step = Tro2(T / k_j, tp, "DCG")
        U_j = np.linalg.matrix_power(U_step, int(k_j))
        rho_j = U_j @ rho0 @ U_j.conj().T
        est += c_j * expval(O, rho_j)

    return np.real_if_close(est).item()


# ================================================================
# Benchmark
# ================================================================

def benchmark_cr_mpf_observable(
    Tlist=None,
    tp=1e-5,
    observable=None,
    initial_state=None,
    klist=(1, 2),
    clist=(-1/3, 4/3),
):
    """
    Benchmark observable-estimation errors for the CR -> Heisenberg example.

    Parameters
    ----------
    Tlist : array-like or None
        List of total evolution times.
    tp : float
        Pulse width.
    observable :
        Can be:
          - None                  -> defaults to "XXII"
          - str                   -> Pauli string, e.g. "XXII", "ZZII"
          - dict                  -> e.g. {0:"X", 1:"X"}
          - ndarray               -> full observable matrix
    initial_state :
        Can be:
          - None                  -> defaults to "0101"
          - str                   -> computational bitstring, e.g. "0101"
          - 1D ndarray            -> state vector
          - 2D ndarray            -> density matrix
    klist, clist :
        MPF parameters. Default is the standard 4th-order two-term MPF:
            M^(2)(T) = (4/3)[S2(T/2)]^2 - (1/3)S2(T)
    """
    if Tlist is None:
        Tlist = np.logspace(-2, 0, 20)
    Tlist = np.asarray(Tlist, dtype=float)

    O = resolve_observable(observable)
    rho_init = resolve_initial_state(initial_state)

    e_T1_obs = []
    e_T2_obs = []
    e_T4_mpf = []

    exp_exact_list = []
    exp_T1_list = []
    exp_T2_list = []
    exp_MPF_list = []

    for T in Tlist:
        U_exact = target_unitary(T)
        rho_exact = U_exact @ rho_init @ U_exact.conj().T
        exp_exact = expval(O, rho_exact)

        U1 = Tro1(T, tp, "DCG")
        rho1 = U1 @ rho_init @ U1.conj().T
        exp1 = expval(O, rho1)

        U2 = Tro2(T, tp, "DCG")
        rho2 = U2 @ rho_init @ U2.conj().T
        exp2 = expval(O, rho2)

        exp4 = mpf_expectation(O, rho_init, klist, clist, T, tp)

        e_T1_obs.append(abs(exp_exact - exp1))
        e_T2_obs.append(abs(exp_exact - exp2))
        e_T4_mpf.append(abs(exp_exact - exp4))

        exp_exact_list.append(exp_exact)
        exp_T1_list.append(exp1)
        exp_T2_list.append(exp2)
        exp_MPF_list.append(exp4)

    return {
        "Tlist": Tlist,
        "tp": float(tp),
        "klist": tuple(klist),
        "clist": tuple(clist),
        "observable": O,
        "rho_init": rho_init,
        "exact": np.asarray(exp_exact_list),
        "T1_DCG": np.asarray(exp_T1_list),
        "T2_DCG": np.asarray(exp_T2_list),
        "MPF_4th": np.asarray(exp_MPF_list),
        "errors": {
            "1st_DCG": np.asarray(e_T1_obs),
            "2nd_DCG": np.asarray(e_T2_obs),
            "4th_MPF": np.asarray(e_T4_mpf),
        },
    }


# ================================================================
# Plot helper
# ================================================================

def plot_cr_mpf_observable(
    bench,
    ylabel=r"$\varepsilon_{\mathrm{MPF}}(O,T)$",
    filename=None,
):
    """
    Plot observable-estimation errors for the CR MPF benchmark.
    """
    Tlist = bench["Tlist"]
    errors = bench["errors"]

    e_T1_obs = errors["1st_DCG"]
    e_T2_obs = errors["2nd_DCG"]
    e_T4_mpf = errors["4th_MPF"]

    ms = 8
    lw = 2.0
    mew = 2

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {
        "1st": "#4C78A8",
        "2nd": "#E45756",
        "4th": "#59A14F",
    }

    ax.plot(
        Tlist, e_T1_obs, "^-",
        color=colors["1st"],
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="1st DCG",
    )

    ax.plot(
        Tlist, e_T2_obs, "^-",
        color=colors["2nd"],
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="2nd DCG",
    )

    ax.plot(
        Tlist, e_T4_mpf, "*-",
        color=colors["4th"],
        markersize=ms + 2,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="4th DCG\n+ MPF",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.legend(fontsize=13, frameon=False, loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.5)

    fig.tight_layout()

    #if filename is not None:
    #    plt.savefig(filename, bbox_inches="tight")

    plt.show()


# ================================================================
# Example run
# ================================================================

if __name__ == "__main__":
    bench = benchmark_cr_mpf_observable(
        Tlist=np.logspace(-2, 0, 20),
        tp=1e-5,
        observable="XXII",
        initial_state="0101",
        klist=(1, 2),
        clist=(-1/3, 4/3),
    )

    plot_cr_mpf_observable(
        bench,
        ylabel=r"$\varepsilon_{\mathrm{MPF}}(X_1X_2,T)$",
        #filename="CR_MPF_observable_error.pdf",
    )