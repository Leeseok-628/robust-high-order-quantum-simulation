import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from construction2_hb import (
    SITE,
    X,
    Y,
    Z,
    I2,
    S1,
    S2,
    P_list,
    P_rev_list,
    free,
    build_tau_klist,
    target_unitary,
)


# ================================================================
# Basic single-qubit states / operators
# ================================================================

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

PAULI_MAP = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}


# ================================================================
# Basic helpers
# ================================================================

def kron_all(op_list):
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out


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
    if len(bitstring) != SITE:
        raise ValueError(f"bitstring must have length {SITE}")

    mapping = {"0": ket0, "1": ket1}

    try:
        state = mapping[bitstring[0]]
        for b in bitstring[1:]:
            state = np.kron(state, mapping[b])
    except KeyError:
        raise ValueError("bitstring must contain only '0' and '1'")

    return state


def local_observable(pauli_map, n_qubits=SITE):
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

    return kron_all(ops)


def observable_from_pauli_string(pauli_string):
    """
    Build an observable from a Pauli string such as:
        "XXII", "ZZII", "XIZY"

    Length must equal SITE.
    """
    pauli_string = pauli_string.upper().replace(" ", "")
    if len(pauli_string) != SITE:
        raise ValueError(
            f"Pauli string must have length {SITE}, got {len(pauli_string)}"
        )

    try:
        ops = [PAULI_MAP[ch] for ch in pauli_string]
    except KeyError:
        raise ValueError("Pauli string must contain only I, X, Y, Z")

    return kron_all(ops)


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
        e.g. {0: "X", 1: "X"} or {0: X, 1: X}
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
    dim = 2 ** SITE
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
    dim = 2 ** SITE

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


def observable_error(O, rho_exact, rho_approx):
    """
    |Tr(O rho_exact) - Tr(O rho_approx)|
    """
    val_exact = np.trace(O @ rho_exact)
    val_approx = np.trace(O @ rho_approx)
    return np.abs(val_exact - val_approx)


# ================================================================
# Scaled robust sequences for hybrid MPF
# ================================================================

def S1_scaled(T, tp, c, Jx, Jy, Jz):
    """
    First-order robust sequence with pulse-stretch factor c.

    This mirrors the pseudocode used in the hybrid MPF construction:
    the free-evolution times are kept fixed by the target coefficients,
    while the Eulerian pulse widths are rescaled through P_list(c, tp).
    """
    tau_klist = build_tau_klist(Jx, Jy, Jz)
    P_klist = P_list(c, tp)

    U = np.eye(2**SITE, dtype=complex)
    for i, P_k in enumerate(P_klist):
        tau_k = tau_klist[i]
        U = (P_k @ free(tau_k * T, tp)) @ U
    return U


def S1_scaled_rev(T, tp, c, Jx, Jy, Jz):
    """
    Time-reversed version of S1_scaled(T, tp, c).
    """
    tau_klist = build_tau_klist(Jx, Jy, Jz)
    P_rev_klist = P_rev_list(c, tp)

    U = np.eye(2**SITE, dtype=complex)
    for i in range(len(P_rev_klist)):
        idx = len(P_rev_klist) - 1 - i
        P_rev_k = P_rev_klist[idx]
        tau_k = tau_klist[idx]
        U = (free(tau_k * T, tp) @ P_rev_k) @ U
    return U


def S2_scaled(T, tp, c, Jx, Jy, Jz):
    """
    Symmetrized second-order robust block built from S1_scaled.

        S2_scaled(T) = S1_scaled(T/2) S1_scaled_rev(T/2)
    """
    U_forw = S1_scaled(T / 2.0, tp, c, Jx, Jy, Jz)
    U_back = S1_scaled_rev(T / 2.0, tp, c, Jx, Jy, Jz)
    return U_forw @ U_back


# ================================================================
# Hybrid MPF state / expectation estimators
# ================================================================

def rho_mpf(rho_in, klist, clist, T, tp, Jx, Jy, Jz):
    """
    Hybrid MPF combination of positive-time robust second-order sequences.

    Uses
        rho_out = sum_j c_j U_j rho_in U_j^\dagger
    with
        U_j = [S2_scaled(T / k_j, tp, c = K / k_j)]^{k_j},
        K = max(klist).
    """
    if len(klist) != len(clist):
        raise ValueError("klist and clist must have the same length")

    if len(klist) == 0:
        raise ValueError("klist must not be empty")

    rho_out = np.zeros_like(rho_in, dtype=complex)
    K = max(klist)

    for k_j, c_j in zip(klist, clist):
        if int(k_j) != k_j or k_j <= 0:
            raise ValueError("All k_j must be positive integers")

        k_j = int(k_j)
        c_scale = K / k_j

        U_step = S2_scaled(T / k_j, tp, c=c_scale, Jx=Jx, Jy=Jy, Jz=Jz)
        U_j = np.linalg.matrix_power(U_step, k_j)
        rho_out += c_j * (U_j @ rho_in @ U_j.conj().T)

    return rho_out


def mpf_expectation(O, rho0, klist, clist, T, tp, Jx, Jy, Jz):
    """
    Returns the hybrid MPF expectation estimator

        sum_j c_j Tr[ O rho_j(T) ],

    where
        rho_j(T) = U_j rho0 U_j^\dagger
        U_j = [S2_scaled(T/k_j, tp, c=K/k_j)]^{k_j}.
    """
    rho_out = rho_mpf(rho0, klist, clist, T, tp, Jx, Jy, Jz)
    return expval(O, rho_out)


# ================================================================
# Benchmark
# ================================================================

def benchmark_algorithm2_mpf_observable(
    Jx=0.10739590438262403,
    Jy=0.25626682612643403,
    Jz=0.2996312103773686,
    Tlist=None,
    tp=1e-5,
    observable=None,
    initial_state=None,
    klist=(1, 2),
    clist=(-1/3, 4/3),
):
    """
    Benchmark observable-estimation errors for the Algorithm 2 /
    anisotropic Heisenberg example.

    Parameters
    ----------
    Jx, Jy, Jz : float
        Target anisotropic Heisenberg couplings.
    Tlist : array-like or None
        Total evolution times.
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
        MPF parameters. Default is the standard two-term 4th-order MPF:
            M^(2)(T) = (4/3)[S2(T/2)]^2 - (1/3)S2(T)
    """
    if Tlist is None:
        Tlist = np.logspace(-2, 0, 20)
    Tlist = np.asarray(Tlist, dtype=float)

    O = resolve_observable(observable)
    rho_init = resolve_initial_state(initial_state)

    e_T1_obs = []
    e_T2_obs = []
    e_T4_obs = []

    exp_exact_list = []
    exp_T1_list = []
    exp_T2_list = []
    exp_MPF_list = []

    for T in Tlist:
        U_exact = target_unitary(T, Jx, Jy, Jz)
        rho_exact = U_exact @ rho_init @ U_exact.conj().T
        exp_exact = expval(O, rho_exact)

        U1 = S1(T, tp, Jx, Jy, Jz)
        rho1 = U1 @ rho_init @ U1.conj().T
        exp1 = expval(O, rho1)

        U2 = S2(T, tp, Jx, Jy, Jz)
        rho2 = U2 @ rho_init @ U2.conj().T
        exp2 = expval(O, rho2)

        exp4 = mpf_expectation(O, rho_init, klist, clist, T, tp, Jx, Jy, Jz)

        e_T1_obs.append(abs(exp_exact - exp1))
        e_T2_obs.append(abs(exp_exact - exp2))
        e_T4_obs.append(abs(exp_exact - exp4))

        exp_exact_list.append(exp_exact)
        exp_T1_list.append(exp1)
        exp_T2_list.append(exp2)
        exp_MPF_list.append(exp4)

    return {
        "Tlist": Tlist,
        "tp": float(tp),
        "Jx": float(Jx),
        "Jy": float(Jy),
        "Jz": float(Jz),
        "klist": tuple(klist),
        "clist": tuple(clist),
        "observable": O,
        "rho_init": rho_init,
        "exact": np.asarray(exp_exact_list),
        "T1_robust": np.asarray(exp_T1_list),
        "T2_robust": np.asarray(exp_T2_list),
        "MPF_4th": np.asarray(exp_MPF_list),
        "errors": {
            "1st_robust": np.asarray(e_T1_obs),
            "2nd_robust": np.asarray(e_T2_obs),
            "4th_robust_MPF": np.asarray(e_T4_obs),
        },
    }


# ================================================================
# Plot helper
# ================================================================

def plot_algorithm2_mpf_observable(
    bench,
    ylabel=r"$\varepsilon_{\mathrm{MPF}}(O,T)$",
    filename=None,
):
    """
    Plot observable-estimation errors for the Algorithm 2 MPF benchmark.
    """
    Tlist = bench["Tlist"]
    errors = bench["errors"]

    e_T1_obs = errors["1st_robust"]
    e_T2_obs = errors["2nd_robust"]
    e_T4_obs = errors["4th_robust_MPF"]

    ms = 8
    mew = 2
    lw = 2.0

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {
        "1st": "#4C78A8",
        "2nd": "#E45756",
        "4th": "#59A14F",
    }

    ax.plot(
        Tlist, e_T1_obs, "o-",
        color=colors["1st"],
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="1st robust",
    )

    ax.plot(
        Tlist, e_T2_obs, "s-",
        color=colors["2nd"],
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="2nd robust",
    )

    ax.plot(
        Tlist, e_T4_obs, "*-",
        color=colors["4th"],
        markersize=ms + 2,
        markerfacecolor="white",
        markeredgewidth=mew,
        lw=lw,
        label="4th robust MPF",
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
    bench = benchmark_algorithm2_mpf_observable(
        Jx=0.10739590438262403,
        Jy=0.25626682612643403,
        Jz=0.2996312103773686,
        Tlist=np.logspace(-2, 0, 20),
        tp=1e-5,
        observable="XXII",
        initial_state="0101",
        klist=(1, 2),
        clist=(-1/3, 4/3),
    )

    plot_algorithm2_mpf_observable(
        bench,
        ylabel=r"$\varepsilon_{\mathrm{MPF}}(X_1X_2,T)$",
        #filename="Algo2_MPF_observable.pdf",
    )