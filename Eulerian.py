from collections import defaultdict
import numpy as np
from itertools import product
import scipy

# ------------------------------------------------
# 1. Eulerian Cycle (Hierholzer's algorithm)
# ------------------------------------------------

def find_eulerian_cycle(graph):
    """
    Find an Eulerian cycle in a directed graph using Hierholzer's algorithm.
    
    Args:
        graph: dict
            Directed adjacency list (node -> list of neighbors)
    
    Returns:
        list or None: Eulerian cycle as a list of nodes, or None if not found.
    """
    if not graph:
        return None

    graph_copy = defaultdict(list)
    edge_count = 0
    for node, neighbors in graph.items():
        graph_copy[node] = neighbors.copy()
        edge_count += len(neighbors)

    # --- Degree checks ---
    in_degree, out_degree = defaultdict(int), defaultdict(int)
    for u in graph:
        out_degree[u] = len(graph[u])
        for v in graph[u]:
            in_degree[v] += 1
    for node in graph:
        if in_degree.get(node, 0) != out_degree.get(node, 0):
            return None

    # --- Connectivity check (simple DFS) ---
    start_node = next(iter(graph))
    visited = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)

    for node in graph:
        if node not in visited and (in_degree.get(node, 0) > 0 or out_degree.get(node, 0) > 0):
            return None

    # --- Hierholzer's algorithm ---
    path, cycle = [start_node], []
    while path:
        node = path[-1]
        if graph_copy[node]:
            path.append(graph_copy[node].pop())
        else:
            cycle.append(path.pop())

    cycle.reverse()
    if len(cycle) != edge_count + 1:
        return None
    return cycle


# ------------------------------------------------
# 2. Pauli Group Utilities
# ------------------------------------------------

PAULI_OPS = ['I', 'X', 'Y', 'Z']

def generate_pauli_group(n):
    """Generate all n-qubit Pauli operators as tuples."""
    return list(product(PAULI_OPS, repeat=n))

def pauli_mult(p1, p2):
    """Element-wise Pauli multiplication (ignoring global phase)."""
    result = []
    for a, b in zip(p1, p2):
        if a == 'I':
            result.append(b)
        elif b == 'I':
            result.append(a)
        elif a == b:
            result.append('I')
        else:
            s = {'X', 'Y', 'Z'}
            s.discard(a)
            s.discard(b)
            result.append(s.pop())  # the remaining Pauli
    return tuple(result)

def compute_transition_paulis(pauli_list):
    """Find P such that p_i * P = p_{i+1}."""
    transitions = []
    for i in range(len(pauli_list) - 1):
        transitions.append(pauli_mult(pauli_list[i], pauli_list[i + 1]))
    return transitions

def generate_cayley_graph(n):
    """
    Generate the Cayley graph for the n-qubit Pauli group
    using generators {X_i, Y_i}.
    """
    paulis = generate_pauli_group(n)
    generators = []
    for i in range(n):
        for axis in ('X', 'Y'):
            g = ['I'] * n
            g[i] = axis
            generators.append(tuple(g))
    
    graph = defaultdict(list)
    for p in paulis:
        for g in generators:
            neighbor = pauli_mult(p, g)
            graph[p].append(neighbor)
    return graph


# ------------------------------------------------
# 3. Pauli Matrix Conversions
# ------------------------------------------------

PAULI_MATRICES = {
    'I': np.array([[1, 0], [0, 1]], dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex)
}

def tensor_product(pauli_string):
    """Compute ⊗ product of a Pauli string, e.g. ('X','Y','I') → X \otimes Y \otimes I."""
    result = PAULI_MATRICES[pauli_string[0]]
    for p in pauli_string[1:]:
        result = np.kron(result, PAULI_MATRICES[p])
    return result

def pauli_strings_to_matrices(pauli_list):
    """Convert a list of Pauli strings to matrices."""
    return [tensor_product(p) for p in pauli_list]


# ------------------------------------------------
# 4. Sum of Single-Qubit Terms
# ------------------------------------------------

def single_term_matrix(pauli_string, pos):
    """Return matrix for term where only qubit[pos] carries its Pauli."""
    n = len(pauli_string)
    ops = []
    for i in range(n):
        ops.append(PAULI_MATRICES[pauli_string[i]] if i == pos else PAULI_MATRICES['I'])
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result

def create_sum_matrix(pauli_string):
    """
    Sum over all non-identity single-qubit Pauli operators.
    e.g., 'IYII' → Y₂.
    """
    n = len(pauli_string)
    result = np.zeros((2**n, 2**n), dtype=complex)
    if all(p == 'I' for p in pauli_string):
        return result
    for i in range(n):
        if pauli_string[i] != 'I':
            result += single_term_matrix(pauli_string, i)
    return result
