import numpy as np
import scipy.linalg

# Change this line if your helper filename is different.
from cr_helper import *
import Eulerian


# ============================================================
# 0. Global model setup
# ============================================================

N_QUBITS = 4
POSITIONS = list(range(N_QUBITS))

# Drift Hamiltonian H_S = sum_i X_i Z_{i+1}
H_S = construct_CR_hamiltonian(N_QUBITS, J=1.0)

# Target Heisenberg Hamiltonian
H_X = construct_hamiltonian_term(X, POSITIONS, n_qubits=N_QUBITS)
H_Y = construct_hamiltonian_term(Y, POSITIONS, n_qubits=N_QUBITS)
H_Z = construct_hamiltonian_term(Z, POSITIONS, n_qubits=N_QUBITS)
H_targ = H_X + H_Y + H_Z


def target_unitary(T):
    return scipy.linalg.expm(-1j * H_targ * T)


# ============================================================
# 1. H_even implementations
# ============================================================

def H_tp(H_S_local, tp):
    """
    Raw finite-width implementation of H_even.
    """
    return U_T(Y24, H_S_local, tp) @ U(Z24, H_S_local, tp)


def H_DCG(H_S_local, tp):
    """
    DCG implementation of H_even.
    """
    return R_Y24_DCG(H_S_local, tp) @ R_Z24_DCG(H_S_local, tp)


# ============================================================
# 2. Trotter constructions
# ============================================================

def _get_blocks(tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    if label == "DCG":
        R_E_G = R_E_DCG(H_S_local, tp)
        R_E_dag_G = R_E_DCG_dag(H_S_local, tp)
        H_G = H_DCG(H_S_local, tp)
    else:
        R_E_G = R_E_tp(H_S_local, tp)
        R_E_dag_G = R_E_tp_dag(H_S_local, tp)
        H_G = H_tp(H_S_local, tp)

    return R_E_G, R_E_dag_G, H_G


def Tro1(tau, tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    R_E_G, R_E_dag_G, H_G = _get_blocks(tp, label, H_S_local)
    ftau = scipy.linalg.expm(-1j * H_S_local * tau)

    P1 = H_G @ R_E_dag_G
    P2 = H_G @ R_E_G @ R_E_G @ H_G
    P3 = H_G @ R_E_dag_G @ H_G
    P4 = H_G

    return P4 @ ftau @ P3 @ ftau @ P2 @ ftau @ P1


def Tro2(tau, tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    R_E_G, R_E_dag_G, H_G = _get_blocks(tp, label, H_S_local)
    ftau = scipy.linalg.expm(-1j * H_S_local * tau / 2.0)

    P1 = H_G @ R_E_dag_G
    P2 = H_G @ R_E_G @ R_E_G @ H_G
    P3 = H_G @ R_E_dag_G @ H_G
    P4 = H_G

    P1_dag = R_E_G @ H_G
    P2_dag = H_G @ R_E_dag_G @ R_E_dag_G @ H_G
    P3_dag = H_G @ R_E_G @ H_G
    P4_dag = H_G

    return (P4 @ ftau @ P3 @ ftau @ P2 @ ftau @ P1) @ (
        P1_dag @ ftau @ P2_dag @ ftau @ P3_dag @ ftau @ P4_dag
    )


def Tro4(tau, tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    u2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    U_2 = Tro2(u2 * tau, tp, label, H_S_local=H_S_local)
    U_2m = Tro2((1.0 - 4.0 * u2) * tau, tp, label, H_S_local=H_S_local)
    return U_2 @ U_2 @ U_2m @ U_2 @ U_2


# ============================================================
# 3. Negative-time / ESCDD ingredients
# ============================================================

Euler1 = [
    ("I", "I", "Y", "I"),
    ("I", "Y", "I", "I"),
    ("I", "I", "Y", "I"),
    ("I", "Y", "I", "I"),
    ("I", "Y", "I", "I"),
    ("I", "I", "Y", "I"),
    ("I", "Y", "I", "I"),
    ("I", "I", "Y", "I"),
]
Euler1_cycle = [Eulerian.create_sum_matrix(pauli) for pauli in Euler1]

cycle = Eulerian.find_eulerian_cycle(Eulerian.generate_cayley_graph(4))
Euler_cycle = Eulerian.compute_transition_paulis(cycle)
Euler_cycle_mat = [Eulerian.create_sum_matrix(pauli) for pauli in Euler_cycle]

pauli_elements = Eulerian.generate_pauli_group(4)
pulse_elements = Eulerian.compute_transition_paulis(pauli_elements)
pulse_elements.append(pauli_elements[-1])
Naive_cycle_mat_4 = [Eulerian.create_sum_matrix(pauli) for pauli in pulse_elements]


# ============================================================
# 4. Refocusing / ESCDD constructions
# ============================================================

def EDD(tau, tp, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    DD = np.eye(2 ** N_QUBITS, dtype=complex)
    ftau = scipy.linalg.expm(-1j * H_S_local * tau)

    for M in Euler1_cycle:
        DD = (U(M, H_S_local, tp) @ ftau) @ DD

    return DD


def EDD_symm(tau, tp, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    DD = np.eye(2 ** N_QUBITS, dtype=complex)
    ftau = scipy.linalg.expm(-1j * H_S_local * tau)

    for M in Euler1_cycle:
        DD = (U(M, H_S_local, tp) @ ftau) @ DD

    for i in range(len(Euler1_cycle)):
        M = Euler1_cycle[len(Euler1_cycle) - 1 - i]
        DD = (ftau @ U_rev(M, H_S_local, tp)) @ DD

    return DD


def refocus_4(tau, tp, H_S_local=None):
    """
    ESCDD for k = 2, without any projection / re-unitarization.
    """
    if H_S_local is None:
        H_S_local = H_S

    assert tau > 0, "tau must be positive"

    ESCDD1_symm = np.eye(2 ** N_QUBITS, dtype=complex)
    EDD_block = EDD_symm(tau, tp, H_S_local=H_S_local)

    for M in Euler_cycle_mat:
        ESCDD1_symm = (U(M, H_S_local, tp) @ EDD_block) @ ESCDD1_symm

    for M in reversed(Euler_cycle_mat):
        ESCDD1_symm = (EDD_block @ U_rev(M, H_S_local, tp)) @ ESCDD1_symm

    return ESCDD1_symm @ scipy.linalg.expm(+1j * H_S_local * tau)


# ============================================================
# 5. Higher-order Trotter with ESCDD
# ============================================================

def Tro2_ESCDD(tau, tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    R_E_G, R_E_dag_G, H_G = _get_blocks(tp, label, H_S_local)

    ftau = refocus_4(-tau / 2.0, tp, H_S_local=H_S_local)

    P1 = H_G @ R_E_dag_G
    P2 = H_G @ R_E_G @ R_E_G @ H_G
    P3 = H_G @ R_E_dag_G @ H_G
    P4 = H_G

    P1_dag = R_E_G @ H_G
    P2_dag = H_G @ R_E_dag_G @ R_E_dag_G @ H_G
    P3_dag = H_G @ R_E_G @ H_G
    P4_dag = H_G

    return (P4 @ ftau @ P3 @ ftau @ P2 @ ftau @ P1) @ (
        P1_dag @ ftau @ P2_dag @ ftau @ P3_dag @ ftau @ P4_dag
    )


def Tro4_ESCDD(tau, tp, label, H_S_local=None):
    if H_S_local is None:
        H_S_local = H_S

    u2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    U_2 = Tro2(u2 * tau, tp, label, H_S_local=H_S_local)
    U_2m = Tro2_ESCDD((1.0 - 4.0 * u2) * tau, tp, label, H_S_local=H_S_local)
    return U_2 @ U_2 @ U_2m @ U_2 @ U_2


# ============================================================
# 6. Benchmark
# ============================================================

def benchmark_construction1_cr(
    Tlist=None,
    tp_list=None,
):
    if Tlist is None:
        Tlist = np.logspace(-2, 0, 13)
    if tp_list is None:
        tp_list = np.array([1e-5, 1e-4, 1e-3], dtype=float)

    Tlist = np.asarray(Tlist, dtype=float)
    tp_list = np.asarray(tp_list, dtype=float)

    results = {}

    for tp in tp_list:
        e_T1_raw, e_T2_raw, e_T4_raw = [], [], []
        e_T1_dcg, e_T2_dcg, e_T4_dcg, e_T4_escdd = [], [], [], []

        for T in Tlist:
            U_target = target_unitary(T)

            U_T1_raw = Tro1(T, tp, "Raw")
            U_T2_raw = Tro2(T, tp, "Raw")
            U_T4_raw = Tro4(T, tp, "Raw")

            U_T1_dcg = Tro1(T, tp, "DCG")
            U_T2_dcg = Tro2(T, tp, "DCG")
            U_T4_dcg = Tro4(T, tp, "DCG")
            U_T4_escdd = Tro4_ESCDD(T, tp, "DCG")

            e_T1_raw.append(gate_error(U_T1_raw, U_target))
            e_T2_raw.append(gate_error(U_T2_raw, U_target))
            e_T4_raw.append(gate_error(U_T4_raw, U_target))

            e_T1_dcg.append(gate_error(U_T1_dcg, U_target))
            e_T2_dcg.append(gate_error(U_T2_dcg, U_target))
            e_T4_dcg.append(gate_error(U_T4_dcg, U_target))
            e_T4_escdd.append(gate_error(U_T4_escdd, U_target))

        results[float(tp)] = {
            "Raw": (
                np.asarray(e_T1_raw),
                np.asarray(e_T2_raw),
                np.asarray(e_T4_raw),
            ),
            "DCG": (
                np.asarray(e_T1_dcg),
                np.asarray(e_T2_dcg),
                np.asarray(e_T4_dcg),
                np.asarray(e_T4_escdd),
            ),
        }

    return {
        "Tlist": Tlist,
        "tp_list": tp_list,
        "results": results,
    }


# ============================================================
# 7. Plot helper
# ============================================================

def plot_cr_benchmark(
    bench,
    tp_target=1e-5,
    filename="CR_Numerics_tp1e-5.pdf",
    ylabel=r"$1-F$",
):
    import matplotlib.pyplot as plt

    Tlist = bench["Tlist"]
    tp_list = bench["tp_list"]
    results = bench["results"]

    ms = 8
    lw = 2.0
    mew = 2

    idx = np.argmin(np.abs(np.array(tp_list) - tp_target))
    tp = float(tp_list[idx])

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {
        "1st": "#4C78A8",
        "2nd": "#E45756",
        "4th": "#59A14F",
        "4th_neg": "#B279A2",
    }

    e_T1, e_T2, e_T4, e_T4_ESCDD = results[tp]["DCG"]
    e_T1_tp, e_T2_tp, e_T4_tp = results[tp]["Raw"]

    ax.plot(
        Tlist, e_T1_tp, "o-", color=colors["1st"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="1st naive"
    )
    ax.plot(
        Tlist, e_T2_tp, "o-", color=colors["2nd"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="2nd naive"
    )
    ax.plot(
        Tlist, e_T4_tp, "o-", color=colors["4th"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="4th naive"
    )

    ax.plot(
        Tlist, e_T1, "^-", color=colors["1st"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="1st DCG"
    )
    ax.plot(
        Tlist, e_T2, "^-", color=colors["2nd"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="2nd DCG"
    )
    ax.plot(
        Tlist, e_T4, "^-", color=colors["4th"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="4th DCG"
    )
    ax.plot(
        Tlist, e_T4_ESCDD, "^-", color=colors["4th_neg"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="4th DCG\n+ neg.-time"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.5)
    ax.legend(fontsize=11, frameon=False, loc="lower right")

    plt.tight_layout()
    #if filename is not None:
    #    plt.savefig(filename, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    bench = benchmark_construction1_cr(
        Tlist=np.logspace(-2, 0, 13),
        tp_list=np.array([1e-5, 1e-4, 1e-3]),
    )

    plot_cr_benchmark(
        bench,
        tp_target=1e-5,
        ylabel=r"$1-F$",
    )