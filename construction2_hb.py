import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ============================================================
# 0. Basic linear-algebra helpers
# ============================================================

def kron_all(op_list):
    """Kronecker product over a list of matrices."""
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out

def gate_error(U, V):
    phase = np.angle(np.trace(U @ V.conj().T))
    return np.linalg.norm(U - np.exp(1j * phase) * V, "fro") ** 2 / (2 * U.shape[0])



# ============================================================
# 1. Basic Pauli operators
# ============================================================

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ============================================================
# 2. Two-site Pauli-pair constructors
# ============================================================

def XX_pair(n_sites, i, j):
    ops = [I2] * n_sites
    ops[i] = X
    ops[j] = X
    return kron_all(ops)


def YY_pair(n_sites, i, j):
    ops = [I2] * n_sites
    ops[i] = Y
    ops[j] = Y
    return kron_all(ops)


def ZZ_pair(n_sites, i, j):
    ops = [I2] * n_sites
    ops[i] = Z
    ops[j] = Z
    return kron_all(ops)


def local_pauli(n_sites, pauli, site):
    ops = [I2] * n_sites
    ops[site] = pauli
    return kron_all(ops)


# ============================================================
# 3. Model construction
# ============================================================

SITE = 4

# Static Hamiltonian H0 = HXX + HYY + HZZ
HXX = np.zeros((2**SITE, 2**SITE), dtype=complex)
HYY = np.zeros((2**SITE, 2**SITE), dtype=complex)
HZZ = np.zeros((2**SITE, 2**SITE), dtype=complex)

for i in range(SITE - 1):
    HXX += XX_pair(SITE, i, i + 1)
    HYY += YY_pair(SITE, i, i + 1)
    HZZ += ZZ_pair(SITE, i, i + 1)

H0 = HXX + HYY + HZZ

# Local operators
X1, Y1, Z1 = local_pauli(SITE, X, 0), local_pauli(SITE, Y, 0), local_pauli(SITE, Z, 0)
X2, Y2, Z2 = local_pauli(SITE, X, 1), local_pauli(SITE, Y, 1), local_pauli(SITE, Z, 1)
X3, Y3, Z3 = local_pauli(SITE, X, 2), local_pauli(SITE, Y, 2), local_pauli(SITE, Z, 2)
X4, Y4, Z4 = local_pauli(SITE, X, 3), local_pauli(SITE, Y, 3), local_pauli(SITE, Z, 3)

# Collective controls on sites 2 and 4
X24 = X2 + X4
Y24 = Y2 + Y4
Z24 = Z2 + Z4

# Coupled operators acting only on (2,4)
X24_f = np.kron(I2, np.kron(X, np.kron(I2, X)))
Y24_f = np.kron(I2, np.kron(Y, np.kron(I2, Y)))
Z24_f = np.kron(I2, np.kron(Z, np.kron(I2, Z)))
I24_f = np.eye(2**SITE, dtype=complex)

# Eulerian group for Algorithm 2
GROUP_24 = [I24_f, X24_f, Z24_f, Y24_f, I24_f, Y24_f, Z24_f, X24_f]


def build_target_hamiltonian(Jx, Jy, Jz):
    return Jx * HXX + Jy * HYY + Jz * HZZ


def build_tau_klist(Jx, Jy, Jz):
    Js = Jx + Jy + Jz
    return 0.25 * np.array([Js, Jx, Jz, Jy, Js, Jy, Jz, Jx], dtype=float)


def target_unitary(T, Jx, Jy, Jz):
    H_targ = build_target_hamiltonian(Jx, Jy, Jz)
    return la.expm(-1j * H_targ * T)


# ============================================================
# 4. Finite-width pulse implementations
# ============================================================

def X24p(tp):
    return la.expm(-1j * ((np.pi / 2) * X24 + H0 * tp))


def Y24p(tp):
    return la.expm(-1j * ((np.pi / 2) * Y24 + H0 * tp))


def X24n(tp):
    return la.expm(-1j * (-(np.pi / 2) * X24 + H0 * tp))


def Y24n(tp):
    return la.expm(-1j * (-(np.pi / 2) * Y24 + H0 * tp))


# ============================================================
# 5. Eulerian pulse lists
# ============================================================

def P_list(c, tp):
    """
    Forward Eulerian pulse sequence on sites (2,4).
    """
    return [
        X24p(c * tp), Y24p(c * tp), X24p(c * tp), Y24p(c * tp),
        Y24p(c * tp), X24p(c * tp), Y24p(c * tp), X24p(c * tp),
    ]


def P_rev_list(c, tp):
    """
    Reverse-time pulse sequence on sites (2,4).
    """
    return [
        X24n(c * tp), Y24n(c * tp), X24n(c * tp), Y24n(c * tp),
        Y24n(c * tp), X24n(c * tp), Y24n(c * tp), X24n(c * tp),
    ]


# ============================================================
# 6. Free evolution under H0
# ============================================================

def free(tau, tp=None):
    """
    Free evolution under H0 for duration tau.
    The tp argument is kept for compatibility with your original code.
    """
    return la.expm(-1j * tau * H0)


# ============================================================
# 7. Algorithm 2 sequences
# ============================================================

def S1(T, tp, Jx, Jy, Jz):
    """
    First-order Eulerian sequence S1(T).
    """
    tau_klist = build_tau_klist(Jx, Jy, Jz)
    c = 1.0
    P_klist = P_list(c, tp)

    U = np.eye(2**SITE, dtype=complex)
    for i, P_k in enumerate(P_klist):
        tau_k = tau_klist[i]
        U = (P_k @ free(tau_k * T, tp)) @ U
    return U


def S1_rev(T, tp, Jx, Jy, Jz):
    """
    Time-reversed version of S1(T).
    """
    tau_klist = build_tau_klist(Jx, Jy, Jz)
    c = 1.0
    P_rev_klist = P_rev_list(c, tp)

    U = np.eye(2**SITE, dtype=complex)
    for i in range(len(P_rev_klist)):
        P_rev_k = P_rev_klist[len(P_rev_klist) - 1 - i]
        tau_k = tau_klist[len(P_rev_klist) - 1 - i]
        U = (free(tau_k * T, tp) @ P_rev_k) @ U
    return U


def S2(T, tp, Jx, Jy, Jz):
    """
    Second-order symmetrized sequence:
        S2(T) = S1(T/2) S1_rev(T/2)
    """
    U_forw = S1(T / 2.0, tp, Jx, Jy, Jz)
    U_back = S1_rev(T / 2.0, tp, Jx, Jy, Jz)
    return U_forw @ U_back


def S4(T, tp, Jx, Jy, Jz):
    """
    Fourth-order sequence with translated middle step.
    """
    tau_klist = build_tau_klist(Jx, Jy, Jz)
    u2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))

    def S2_M(T_local, tp_local):
        c = -(1.0 - 4.0 * u2) / u2
        p_list = P_list(c, tp_local)
        p_rev_list = P_rev_list(c, tp_local)

        def p_phi(j):
            U = np.eye(2**SITE, dtype=complex)
            for i in range(j - 1, -1, -1):
                U = U @ p_list[i]
            for i in range(len(p_list) - 1, j, -1):
                U = U @ p_list[i]
            return U

        def p_rev_phi(j):
            U = np.eye(2**SITE, dtype=complex)
            for i in range(j + 1, len(p_list)):
                U = U @ p_rev_list[i]
            for i in range(0, j):
                U = U @ p_rev_list[i]
            return U

        U_forw = np.eye(2**SITE, dtype=complex)
        for i in range(len(p_list)):
            tau_k = tau_klist[i]
            U_forw = (p_rev_phi(i) @ free(tau_k * T_local / 2.0, tp_local)) @ U_forw

        U_back = np.eye(2**SITE, dtype=complex)
        for i in range(len(p_list)):
            tau_k = tau_klist[len(p_list) - 1 - i]
            U_back = (free(tau_k * T_local / 2.0, tp_local) @ p_phi(len(p_list) - 1 - i)) @ U_back

        return U_forw @ U_back

    return (
        S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2_M((1.0 - 4.0 * u2) * T, tp)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
    )


def S4_raw(T, tp, Jx, Jy, Jz):
    """
    Fourth-order composition without translated middle step.
    """
    u2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    return (
        S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2((1.0 - 4.0 * u2) * T, tp, Jx, Jy, Jz)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
        @ S2(u2 * T, tp, Jx, Jy, Jz)
    )


# ============================================================
# 8. Benchmark driver
# ============================================================


def benchmark_algorithm2_heisenberg(
    Jx=0.10739590438262403,
    Jy=0.25626682612643403,
    Jz=0.2996312103773686,
    Tlist=None,
    tp_values=None,
    metric="paper",
):
    """
    Returns a dict with the same structure as your original notebook:
        results[tp]["Tlist"]
        results[tp]["e1"]
        results[tp]["e2"]
        results[tp]["e4"]
        results[tp]["e4_raw"]
    """
    if Tlist is None:
        Tlist = np.logspace(-1.2, 0, 20)
    if tp_values is None:
        tp_values = [1e-3, 1e-4, 1e-5]

    Tlist = np.asarray(Tlist, dtype=float)
    tp_values = np.asarray(tp_values, dtype=float)

    results = {}

    for tp in tp_values:
        e1, e2, e4, e4_raw = [], [], [], []

        for T in Tlist:
            U_id = target_unitary(T, Jx, Jy, Jz)

            e1.append(gate_error(U_id, S1(T, tp, Jx, Jy, Jz)))
            e2.append(gate_error(U_id, S2(T, tp, Jx, Jy, Jz)))
            e4.append(gate_error(U_id, S4(T, tp, Jx, Jy, Jz)))
            e4_raw.append(gate_error(U_id, S4_raw(T, tp, Jx, Jy, Jz)))

        results[float(tp)] = {
            "Tlist": Tlist,
            "e1": np.asarray(e1),
            "e2": np.asarray(e2),
            "e4": np.asarray(e4),
            "e4_raw": np.asarray(e4_raw),
        }

    return results


# ============================================================
# 9. Plot helper
# ============================================================

def plot_algorithm2_panel(
    results,
    tp_target=1e-4,
    filename="Algo2_Numerics_tp1e-4.pdf",
    ylabel=r"$1-F$",
):
    ms = 8
    lw = 2.0
    mew = 2

    tp_keys = np.array(list(results.keys()), dtype=float)
    tp = float(tp_keys[np.argmin(np.abs(tp_keys - tp_target))])

    Tlist_plot = results[tp]["Tlist"]
    e1 = results[tp]["e1"]
    e2 = results[tp]["e2"]
    e4 = results[tp]["e4"]
    e4_raw = results[tp]["e4_raw"]

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {
        "1st": "#4C78A8",
        "2nd": "#E45756",
        "4th": "#59A14F",
    }

    ax.plot(
        Tlist_plot, e1, "^-", color=colors["1st"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="1st robust"
    )
    ax.plot(
        Tlist_plot, e2, "^-", color=colors["2nd"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="2nd robust"
    )
    ax.plot(
        Tlist_plot, e4, "^-", color=colors["4th"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="4th robust"
    )

    ax.plot(
        Tlist_plot, e4_raw, "o-", color=colors["4th"], markersize=ms,
        markerfacecolor="white", markeredgewidth=mew, lw=lw,
        label="4th naive"
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


# ============================================================
# 10. One-call notebook/demo entrypoint
# ============================================================

def run_algorithm2_demo(
    Jx=0.10739590438262403,
    Jy=0.25626682612643403,
    Jz=0.2996312103773686,
    tp_target=1e-4,
    Tlist=None,
    tp_values=None,
    filename="Algo2_Numerics_tp1e-4.pdf",
    return_results=False,
):
    """
    Benchmark + plot in one call.
    """
    results = benchmark_algorithm2_heisenberg(
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Tlist=Tlist,
        tp_values=tp_values,
    )

    plot_algorithm2_panel(
        results,
        tp_target=tp_target,
        filename=filename,
        ylabel=r"$1-F$",
    )

    if return_results:
        return results


# ============================================================
# 11. Demo
# ============================================================

if __name__ == "__main__":
    run_algorithm2_demo(
        Jx=0.10739590438262403,
        Jy=0.25626682612643403,
        Jz=0.2996312103773686,
        tp_target=1e-4,
        Tlist=np.logspace(-1.2, 0, 20),
        tp_values=[1e-3, 1e-4, 1e-5],
        #filename="Algo2_Numerics_tp1e-4.pdf",
        return_results=False,
    )