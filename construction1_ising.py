import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import Eulerian


# ============================================================
# 0. Basic Pauli operators
# ============================================================

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_on_site(n_qubits, pauli, site):
    ops = [I2] * n_qubits
    ops[site] = pauli
    return kron_all(ops)


# ============================================================
# 1. n = 3 Ising setup
# ============================================================

n_qubits = 3

X1, X2, X3 = (pauli_on_site(n_qubits, X, i) for i in range(3))
Y1, Y2, Y3 = (pauli_on_site(n_qubits, Y, i) for i in range(3))
Z1, Z2, Z3 = (pauli_on_site(n_qubits, Z, i) for i in range(3))

# Native Hamiltonian H_0
H_0 = Z1 @ Z2 + Z1 @ Z3 + Z2 @ Z3

# Available controls
X12 = X1 + X2
X13 = X1 + X3
X23 = X2 + X3


# ============================================================
# 2. Primitive finite-width pulses
# ============================================================

def rectangular_pulse(H_ctrl, tp):
    """
    Primitive finite-width pulse:
        W_f = exp[-i ( (pi/2) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * (np.pi / 2 * H_ctrl + tp * H_0))


def reversed_rectangular_pulse(H_ctrl, tp):
    """
    Reversed primitive pulse:
        W_f^rev = exp[-i ( -(pi/2) H_ctrl + tp H_0 )]
    """
    return scipy.linalg.expm(-1j * (-np.pi / 2 * H_ctrl + tp * H_0))


def stretched_rectangular_pulse(H_ctrl, c, tp, reversed_pulse=False):
    """
    Stretched pulse:
        W_f(c tp) or W_f^rev(c tp)
    """
    sign = -1.0 if reversed_pulse else +1.0
    return scipy.linalg.expm(-1j * (sign * np.pi / 2 * H_ctrl + c * tp * H_0))


def free_evolution(tau):
    return scipy.linalg.expm(-1j * H_0 * tau)


# ============================================================
# 3. Error metrics
# ============================================================

def gate_error(U, V):
    """
    The error metric:
        || U - e^{i phi} V ||_F^2 / (2 d)
        
    This is equivalent to the paper metric:
        1 - |Tr(U^\dagger V)| / d
    """
    phase = np.angle(np.trace(U @ V.conj().T))
    return np.linalg.norm(U - np.exp(1j * phase) * V, "fro") ** 2 / (2 * U.shape[0])


# ============================================================
# 4. Target Construction-1 Ising model
# ============================================================

def target_hamiltonian(J12, J13, J23):
    return (
        J12 * (Z1 @ Z2)
        + J13 * (Z1 @ Z3)
        + J23 * (Z2 @ Z3)
    )


def target_unitary(J12, J13, J23, T=1.0):
    return scipy.linalg.expm(-1j * target_hamiltonian(J12, J13, J23) * T)


def construction1_nonzero_delays(J12, J13, J23, T=1.0, J_native=1.0):
    """
    Returns:
        [tau_12, tau_23, tau_34]
    """
    return -(T / (2.0 * J_native)) * np.array(
        [
            J23 + J13,
            J12 + J23,
            J12 + J13,
        ],
        dtype=float,
    )


# ============================================================
# 5. Augmented Cayley-graph helpers
# ============================================================

def build_augmented_word(vertices, generators):
    """
    Build augmented Eulerian word on a user-specified Cayley graph.

    Edge labels:
      generator edge -> corresponding generator tuple
      self-loop      -> "L"
      exit edge      -> "E"
      dummy return   -> "DUMMY"
    """
    identity = tuple("I" for _ in vertices[0])
    terminal = ("TERMINAL",)

    graph = {v: [Eulerian.pauli_mult(v, g) for g in generators] for v in vertices}
    aug_graph = {v: list(nbrs) for v, nbrs in graph.items()}

    for v in vertices:
        if v != identity:
            aug_graph[v].append(v)

    aug_graph[identity].append(terminal)
    aug_graph[terminal] = [identity]

    cycle = Eulerian.find_eulerian_cycle(aug_graph)
    if cycle is None:
        raise RuntimeError("Failed to find Eulerian cycle on augmented graph.")

    def edge_label(u, v):
        if u == identity and v == terminal:
            return "E"
        if u == terminal and v == identity:
            return "DUMMY"
        if u == v and u != identity:
            return "L"
        for g in generators:
            if Eulerian.pauli_mult(u, g) == v:
                return g
        raise RuntimeError(f"Could not label edge {(u, v)}.")

    labels = [edge_label(u, v) for u, v in zip(cycle[:-1], cycle[1:])]
    k = labels.index("DUMMY")
    return labels[k + 1:] + labels[:k]


# ============================================================
# 6. Short first-order DCG for Ising
#    (used directly for q=1 and also inside q=2)
# ============================================================

def x_pulses_for_pair(pair):
    Xs = [X1, X2, X3]
    a, b = pair
    return Xs[a], Xs[b]


def short_first_order_dcg(target_ctrl, pair, tp, inverse=False):
    """
    First-order DCG for the Ising model.

    pair:
        active pair for the Eulerian walk

    inverse=False:
        W_f^[1]

    inverse=True:
        corresponding block for W^\dagger
    """
    Xa, Xb = x_pulses_for_pair(pair)

    if not inverse:
        I_W = reversed_rectangular_pulse(target_ctrl, tp) @ rectangular_pulse(target_ctrl, tp)
        W_edge = stretched_rectangular_pulse(target_ctrl, 2.0, tp, reversed_pulse=False)
    else:
        I_W = rectangular_pulse(target_ctrl, tp) @ reversed_rectangular_pulse(target_ctrl, tp)
        W_edge = stretched_rectangular_pulse(target_ctrl, 2.0, tp, reversed_pulse=True)

    # Time-ordered version:
    # rightmost factor in the paper acts first.
    seq = [
        rectangular_pulse(Xa, tp),
        I_W,
        rectangular_pulse(Xb, tp),
        I_W,
        rectangular_pulse(Xa, tp),
        I_W,
        rectangular_pulse(Xb, tp),
        rectangular_pulse(Xb, tp),
        rectangular_pulse(Xa, tp),
        rectangular_pulse(Xb, tp),
        rectangular_pulse(Xa, tp),
        W_edge,
    ]

    U_tot = np.eye(2 ** n_qubits, dtype=complex)
    for block in seq:
        U_tot = block @ U_tot
    return U_tot


def dcg1_sequence(target_ctrl, pair, tp):
    """
    q = 1 block:
    use the first-order Ising DCG directly.
    """
    return short_first_order_dcg(target_ctrl, pair, tp, inverse=False)


# ============================================================
# 7. Full active-pair Cayley graph for q = 2
#    outer layer only
# ============================================================

GAMMA_ACTIVE_PAIR = [
    ("X", "I"),
    ("Y", "I"),
    ("I", "X"),
    ("I", "Y"),
]

GROUP_ACTIVE_PAIR = Eulerian.generate_pauli_group(2)
WORD_ACTIVE_PAIR = build_augmented_word(GROUP_ACTIVE_PAIR, GAMMA_ACTIVE_PAIR)


def generator_map_for_pair_xy(pair):
    Xs = [X1, X2, X3]
    Ys = [Y1, Y2, Y3]
    a, b = pair
    return {
        ("X", "I"): Xs[a],
        ("Y", "I"): Ys[a],
        ("I", "X"): Xs[b],
        ("I", "Y"): Ys[b],
    }


# ============================================================
# 8. q = 2 block:
#    outer full-group Eulerian path
#    inner generator blocks = short q=1 DCGs
# ============================================================

def dcg2_sequence(target_ctrl, pair, tp):
    """
    Hybrid version:

      outer layer:
          full active-pair augmented Cayley graph

      inner generator blocks:
          short first-order DCGs
    """
    gen_map = generator_map_for_pair_xy(pair)

    gen_blocks_1 = {
        label: short_first_order_dcg(H_ctrl, pair, tp, inverse=False)
        for label, H_ctrl in gen_map.items()
    }

    W1 = short_first_order_dcg(target_ctrl, pair, tp, inverse=False)
    W1_dag = short_first_order_dcg(target_ctrl, pair, tp, inverse=True)

    # The q=1 block has total duration 16 tp,
    # W_f^[1](sqrt(2) tau_1) is modeled by tp -> sqrt(2) tp.
    W1_stretched = short_first_order_dcg(
        target_ctrl,
        pair,
        np.sqrt(2.0) * tp,
        inverse=False,
    )

    I_W_1 = W1_dag @ W1_stretched
    W_star_1 = W1 @ W1_dag @ W1

    U_tot = np.eye(2 ** n_qubits, dtype=complex)
    for label in WORD_ACTIVE_PAIR:
        if label == "L":
            block = I_W_1
        elif label == "E":
            block = W_star_1
        else:
            block = gen_blocks_1[label]
        U_tot = block @ U_tot

    return U_tot


# ============================================================
# 9. Negative-time blocks
# ============================================================

def negative_time_block_naive(tau_val, tp):
    """
    Implements e^{+i H_0 |tau_val|} for tau_val < 0
    using the same refocusing-style block as your notebook.
    """
    X1_gate = rectangular_pulse(X1, tp)
    X2_gate = rectangular_pulse(X2, tp)

    return (
        X2_gate
        @ free_evolution(-tau_val)
        @ X1_gate
        @ free_evolution(-tau_val)
        @ X2_gate
        @ free_evolution(-tau_val)
        @ X1_gate
    )


def effective_free_naive(tau_val, tp):
    return free_evolution(tau_val) if tau_val >= 0 else negative_time_block_naive(tau_val, tp)


def negative_time_block_q1(tau_val, tp):
    DCG1_q1 = dcg1_sequence(X1, (0, 1), tp)
    DCG2_q1 = dcg1_sequence(X2, (0, 1), tp)

    return (
        DCG2_q1
        @ free_evolution(-tau_val)
        @ DCG1_q1
        @ free_evolution(-tau_val)
        @ DCG2_q1
        @ free_evolution(-tau_val)
        @ DCG1_q1
    )


def effective_free_q1(tau_val, tp):
    return free_evolution(tau_val) if tau_val >= 0 else negative_time_block_q1(tau_val, tp)


def negative_time_block_q2(tau_val, tp):
    DCG1_q2 = dcg2_sequence(X1, (0, 1), tp)
    DCG2_q2 = dcg2_sequence(X2, (0, 1), tp)

    return (
        DCG2_q2
        @ free_evolution(-tau_val)
        @ DCG1_q2
        @ free_evolution(-tau_val)
        @ DCG2_q2
        @ free_evolution(-tau_val)
        @ DCG1_q2
    )


def effective_free_q2(tau_val, tp):
    return free_evolution(tau_val) if tau_val >= 0 else negative_time_block_q2(tau_val, tp)


# ============================================================
# 10. Full Construction-1 Ising sequences
# ============================================================

def full_sequence_naive(J12, J13, J23, tp, T=1.0, taus=None):
    if taus is None:
        tau12, tau23, tau34 = construction1_nonzero_delays(J12, J13, J23, T=T)
    else:
        tau12, tau23, tau34 = np.asarray(taus, dtype=float)

    X12_gate = rectangular_pulse(X12, tp)
    X23_gate = rectangular_pulse(X23, tp)

    return (
        X23_gate
        @ effective_free_naive(tau34, tp)
        @ X12_gate
        @ effective_free_naive(tau23, tp)
        @ X23_gate
        @ effective_free_naive(tau12, tp)
        @ X12_gate
    )


def full_sequence_q1(J12, J13, J23, tp, T=1.0, taus=None):
    if taus is None:
        tau12, tau23, tau34 = construction1_nonzero_delays(J12, J13, J23, T=T)
    else:
        tau12, tau23, tau34 = np.asarray(taus, dtype=float)

    DCG12_q1 = dcg1_sequence(X12, (0, 1), tp)
    DCG23_q1 = dcg1_sequence(X23, (1, 2), tp)

    return (
        DCG23_q1
        @ effective_free_q1(tau34, tp)
        @ DCG12_q1
        @ effective_free_q1(tau23, tp)
        @ DCG23_q1
        @ effective_free_q1(tau12, tp)
        @ DCG12_q1
    )


def full_sequence_q2(J12, J13, J23, tp, T=1.0, taus=None):
    if taus is None:
        tau12, tau23, tau34 = construction1_nonzero_delays(J12, J13, J23, T=T)
    else:
        tau12, tau23, tau34 = np.asarray(taus, dtype=float)

    DCG12_q2 = dcg2_sequence(X12, (0, 1), tp)
    DCG23_q2 = dcg2_sequence(X23, (1, 2), tp)

    return (
        DCG23_q2
        @ effective_free_q2(tau34, tp)
        @ DCG12_q2
        @ effective_free_q2(tau23, tp)
        @ DCG23_q2
        @ effective_free_q2(tau12, tp)
        @ DCG12_q2
    )


# ============================================================
# 11. Benchmark helpers
# ============================================================

def slope_loglog(x, y, floor=1e-18):
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), floor)
    return np.polyfit(np.log10(x), np.log10(y), 1)[0]


def benchmark_full_ising(
    J12=-0.547,
    J13=0.925,
    J23=-0.747,
    T=1.0,
    tp_list=None,
    taus=None,
):
    if tp_list is None:
        tp_list = np.logspace(-4, -1, 20)

    tp_list = np.asarray(tp_list, dtype=float)
    U_targ = target_unitary(J12, J13, J23, T=T)

    
    e_naive, e_q1, e_q2 = [], [], []

    for tp in tp_list:
        U_naive = full_sequence_naive(J12, J13, J23, tp, T=T, taus=taus)
        U_q1 = full_sequence_q1(J12, J13, J23, tp, T=T, taus=taus)
        U_q2 = full_sequence_q2(J12, J13, J23, tp, T=T, taus=taus)

        e_naive.append(gate_error(U_naive, U_targ))
        e_q1.append(gate_error(U_q1, U_targ))
        e_q2.append(gate_error(U_q2, U_targ))

    e_naive = np.asarray(e_naive)
    e_q1 = np.asarray(e_q1)
    e_q2 = np.asarray(e_q2)

    return {
        "tp": tp_list,
        "e_naive": e_naive,
        "e_q1": e_q1,
        "e_q2": e_q2,
        "slope_naive": slope_loglog(tp_list, e_naive),
        "slope_q1": slope_loglog(tp_list, e_q1),
        "slope_q2": slope_loglog(tp_list, e_q2),
    }


# ============================================================
# 12. Plot helper
# ============================================================

def plot_ising_benchmark(
    data,
    filename="Ising_Numerics_q2.pdf",
    ylabel=r"$1 - F$",
):
    ms = 8
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        data["tp"], data["e_naive"], "o-",
        color="#4C78A8",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=1.5,
        lw=2.0,
        label=r"Naive $q=0$",
    )

    ax.plot(
        data["tp"], data["e_q1"], "^-",
        color="#E45756",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=1.5,
        lw=2.0,
        label=r"DCG $q=1$",
    )

    ax.plot(
        data["tp"], data["e_q2"], "d-",
        color="#2E2E2E",
        markersize=ms,
        markerfacecolor="white",
        markeredgewidth=1.5,
        lw=2.0,
        label=r"DCG $q=2$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_p$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(fontsize=15, frameon=False, loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.5)

    plt.tight_layout()
    #if filename is not None:
    #    plt.savefig(filename, bbox_inches="tight")
    plt.show()


# ============================================================
# 13. Demo
# ============================================================

if __name__ == "__main__":
    J12, J13, J23 = -0.547, 0.925, -0.747
    taus = construction1_nonzero_delays(J12, J13, J23, T=1.0)

    print(f"J12 = {J12:.3f}, J13 = {J13:.3f}, J23 = {J23:.3f}")
    print("taus =", taus)
    print("len(WORD_ACTIVE_PAIR) =", len(WORD_ACTIVE_PAIR))

    data = benchmark_full_ising(
        J12=J12,
        J13=J13,
        J23=J23,
        T=1.0,
        taus=taus,
    )

    print(
        f"Scaling slopes: naive = {data['slope_naive']:.2f}, "
        f"q=1 = {data['slope_q1']:.2f}, "
        f"q=2 = {data['slope_q2']:.2f}"
    )

    plot_ising_benchmark(
        data,
        #filename="Ising_Numerics_q2.pdf",
        ylabel=r"$1 - F$",
    )