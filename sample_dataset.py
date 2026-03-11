"""
Microstructure Dataset Sampler - Extended Parameter Space
=========================================================
Generates 1000 microstructure samples with the following parameter ranges:
  - Nodes per edge: 1 or 2 (random)
  - Node width d: rand(0.2, 0.4) * L where L=1.0 (frame side length, normalized)
  - Adjacency matrix size: variable N×N (N = total nodes, 4 to 8)
  - Node degree: [1, 2] (each node must have at least 1 connection)
  - Single connected graph enforced via BFS
  - Connection types: straight_line or bezier_curve only

Output: Output/samples/
  images/      - 128×128 PNG microstructure images
  radars/      - Polar property radar charts
  dataset.json - Metadata with parameters and computed properties
"""

import os
import sys
import json
import math
import random
import argparse
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ── local imports ────────────────────────────────────────────────────────────
from generate_microstructure import render_microstructure
from homogenize import load_and_reconstruct, homogenize_elastic, homogenize_thermal


# ═════════════════════════════════════════════════════════════════════════════
# 1. Topology Generation
# ═════════════════════════════════════════════════════════════════════════════

EDGES = ["E1", "E2", "E3", "E4"]

def sample_nodes_on_edge(n_nodes):
    """
    Sample n_nodes positions on a single edge in (0.1, 0.9).
    If n_nodes == 2, ensure the two positions are at least 0.15 apart.
    Returns sorted list of positions.
    """
    if n_nodes == 1:
        return [round(random.uniform(0.1, 0.9), 4)]
    else:
        for _ in range(500):
            a = random.uniform(0.1, 0.9)
            b = random.uniform(0.1, 0.9)
            if abs(a - b) >= 0.15:
                return sorted([round(a, 4), round(b, 4)])
        # Fallback: place at 0.3 and 0.7
        return [0.3, 0.7]


def build_nodes_config():
    """
    Randomly decide how many nodes (1 or 2) to place on each edge,
    then sample their positions.
    Returns:
        nodes_config: dict  {edge_id: [pos0, pos1, ...]}
        node_list:    list of (edge_id, node_idx) in a fixed order
    """
    nodes_config = {}
    node_list = []  # ordered list of (edge, idx) for matrix indexing
    for edge in EDGES:
        n = random.choice([1, 2])
        positions = sample_nodes_on_edge(n)
        nodes_config[edge] = positions
        for i in range(n):
            node_list.append((edge, i))
    return nodes_config, node_list


def node_center(nodes_config, edge, idx):
    """Return (x, y) center of a node in normalized [0,1]^2 coords."""
    pos = nodes_config[edge][idx]
    if edge == "E1":   return (pos, 0.0)
    if edge == "E2":   return (1.0, pos)
    if edge == "E3":   return (pos, 1.0)
    if edge == "E4":   return (0.0, pos)


def edge_length(nodes_config, node_a, node_b):
    """Euclidean distance between two node centers."""
    ca = node_center(nodes_config, *node_a)
    cb = node_center(nodes_config, *node_b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def is_valid_graph(adj, n):
    """
    Checks:
      1. Each node degree is exactly in [1, 2].
      2. Graph is single-connected (BFS).
    """
    degrees = [sum(adj[i]) for i in range(n)]
    if any(d < 1 or d > 2 for d in degrees):
        return False
    # BFS
    visited = {0}
    queue = [0]
    while queue:
        cur = queue.pop()
        for nb in range(n):
            if adj[cur][nb] and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


def sample_adjacency(n, max_attempts=50000):
    """
    Sample a random valid N×N symmetric adjacency matrix satisfying:
      - degree in [1, 2] for every node
      - single connected component
    Returns adj list of lists, or None on failure.
    """
    for _ in range(max_attempts):
        adj = [[0] * n for _ in range(n)]
        # Randomly add edges
        p = random.uniform(0.3, 0.85)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        if is_valid_graph(adj, n):
            return adj
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Width Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_widths(nodes_config, node_list, adj):
    """
    For each node, assign width d = rand(0.2, 0.4) * 1.0
    (l = frame side length = 1.0 in normalized coordinates).
    Then clamp so that [t - d/2, t + d/2] stays within (0, 1).
    Returns: dict {(edge, idx): d}
    """
    n = len(node_list)
    widths = {}
    for i, (edge, idx) in enumerate(node_list):
        d_raw = random.uniform(0.2, 0.4)
        pos = nodes_config[edge][idx]
        # Clamp: pos - d/2 >= 0.01 and pos + d/2 <= 0.99
        max_d = 2 * min(pos - 0.01, 0.99 - pos)
        d = min(d_raw, max_d)
        d = max(d, 0.01)  # safety floor
        widths[(edge, idx)] = round(d, 4)
    return widths


# ═════════════════════════════════════════════════════════════════════════════
# 3. Build render params
# ═════════════════════════════════════════════════════════════════════════════

CONN_TYPES = ["straight_line", "bezier_curve"]


def build_render_params(nodes_config, node_list, adj, widths):
    """
    Construct the params dict expected by render_microstructure().
    Format:
      {
        "nodes": {"E1": [[t, d], ...], ...},
        "connections": [{"start": [edge, idx], "end": [edge, idx], "type": ...}, ...]
      }
    """
    n = len(node_list)
    # nodes section
    nodes_dict = {}
    for edge in EDGES:
        nodes_dict[edge] = []
    for (edge, idx), d in widths.items():
        pos = nodes_config[edge][idx]
        # Ensure slot exists
        while len(nodes_dict[edge]) <= idx:
            nodes_dict[edge].append(None)
        nodes_dict[edge][idx] = [pos, d]

    # connections
    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                c_type = random.choice(CONN_TYPES)
                connections.append({
                    "start": list(node_list[i]),
                    "end": list(node_list[j]),
                    "type": c_type
                })

    return {"nodes": nodes_dict, "connections": connections}


# ═════════════════════════════════════════════════════════════════════════════
# 4. Radar Chart
# ═════════════════════════════════════════════════════════════════════════════

def compute_polar_E(C_eff, num_pts=360):
    """Young's modulus E(theta) polar curve."""
    try:
        S = np.linalg.inv(C_eff)
    except np.linalg.LinAlgError:
        return None, None
    S11, S22 = S[0, 0], S[1, 1]
    S12, S66 = S[0, 1], S[2, 2]
    S16, S26 = S[0, 2], S[1, 2]
    thetas = np.linspace(0, 2 * np.pi, num_pts)
    E_vals = []
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        inv_E = (S11 * c**4 + S22 * s**4
                 + (2 * S12 + S66) * c**2 * s**2
                 + 2 * S16 * c**3 * s + 2 * S26 * c * s**3)
        E_vals.append(1.0 / inv_E if inv_E > 1e-15 else np.nan)
    thetas = np.append(thetas, thetas[0])
    E_vals = np.append(E_vals, E_vals[0])
    return thetas, E_vals


def compute_polar_kappa(kappa_eff, num_pts=360):
    """Thermal conductivity kappa(theta) polar curve."""
    k11, k22, k12 = kappa_eff[0, 0], kappa_eff[1, 1], kappa_eff[0, 1]
    thetas = np.linspace(0, 2 * np.pi, num_pts)
    kappa_vals = []
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        kappa_vals.append(k11 * c**2 + k22 * s**2 + 2 * k12 * c * s)
    thetas = np.append(thetas, thetas[0])
    kappa_vals = np.append(kappa_vals, kappa_vals[0])
    return thetas, np.array(kappa_vals)


def plot_radar(C_eff, kappa_eff, out_path, title=""):
    """Generate and save combined normalized polar radar chart."""
    thetas_E, E_vals = compute_polar_E(C_eff)
    thetas_k, k_vals = compute_polar_kappa(kappa_eff)

    if thetas_E is None:
        return

    max_E = np.nanmax(E_vals) if np.any(~np.isnan(E_vals)) else 1.0
    max_k = np.nanmax(k_vals) if np.any(~np.isnan(k_vals)) else 1.0

    E_norm = E_vals / max_E if max_E > 0 else E_vals
    k_norm = k_vals / max_k if max_k > 0 else k_vals

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(thetas_E, E_norm, linewidth=2, color='royalblue',
            label=f"E(θ)  max={max_E:.3f}")
    ax.fill(thetas_E, E_norm, alpha=0.15, color='royalblue')

    ax.plot(thetas_k, k_norm, linewidth=2, color='crimson',
            label=f"κ(θ)  max={max_k:.3f}")
    ax.fill(thetas_k, k_norm, alpha=0.12, color='crimson')

    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='dimgray')
    ax.set_rlabel_position(22.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(title, pad=18, fontsize=10, weight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Single Sample Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_one_sample(seed=None):
    """
    Generate one valid sample: topology → image → homogenization → radar.
    Returns dict with all data, or raises exception on failure.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(200):
        nodes_config, node_list = build_nodes_config()
        n = len(node_list)

        adj = sample_adjacency(n)
        if adj is None:
            continue  # Retry with new node config

        widths = compute_widths(nodes_config, node_list, adj)
        params = build_render_params(nodes_config, node_list, adj, widths)
        return params, nodes_config, node_list, adj, widths

    raise RuntimeError("Failed to generate a valid sample after 200 topology attempts.")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Dataset Builder
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(num_samples=1000, out_dir="Output/samples", img_size=(128, 128)):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    radar_dir = os.path.join(out_dir, "radars")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)

    records = []
    failed = 0

    print(f"{'='*60}")
    print(f" TMMStructures - Sampling {num_samples} microstructures")
    print(f" Output → {os.path.abspath(out_dir)}")
    print(f"{'='*60}")

    for i in range(num_samples):
        base = f"sample_{i:04d}"
        img_path   = os.path.join(img_dir,   f"{base}.png")
        radar_path = os.path.join(radar_dir, f"{base}_radar.png")

        try:
            params, nodes_config, node_list, adj, widths = generate_one_sample(seed=1000 + i)

            # ── Render image ──────────────────────────────────────────
            render_microstructure(params, size=img_size, out_path=img_path)

            # ── Homogenization ────────────────────────────────────────
            density = load_and_reconstruct(img_path, invert=True)
            nely, nelx = density.shape
            volume_fraction = float(np.mean(density))

            C_eff     = homogenize_elastic(nelx, nely, density, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0)
            kappa_eff = homogenize_thermal(nelx, nely, density, k0=1.0, kmin=1e-9, penal=3.0)

            # ── Radar chart ───────────────────────────────────────────
            plot_radar(C_eff, kappa_eff, radar_path, title=base)

            # ── Record ────────────────────────────────────────────────
            # Serialize adj (list of lists is already JSON-serialisable)
            widths_serial = {f"{e}_{idx}": d for (e, idx), d in widths.items()}

            record = {
                "id":               base,
                "image_path":       img_path,
                "radar_path":       radar_path,
                "volume_fraction":  round(volume_fraction, 6),
                "schema": {
                    "nodes_config": {e: nodes_config[e] for e in EDGES},
                    "node_list":    [[e, idx] for (e, idx) in node_list],
                    "adj_matrix":   adj,
                    "widths":       widths_serial,
                    "connections":  params["connections"],
                },
                "C_eff":     C_eff.tolist(),
                "kappa_eff": kappa_eff.tolist(),
                # Handy scalar summaries
                "C11": float(C_eff[0, 0]), "C12": float(C_eff[0, 1]),
                "C22": float(C_eff[1, 1]), "C66": float(C_eff[2, 2]),
                "C16": float(C_eff[0, 2]), "C26": float(C_eff[1, 2]),
                "k11": float(kappa_eff[0, 0]), "k12": float(kappa_eff[0, 1]),
                "k22": float(kappa_eff[1, 1]),
            }
            records.append(record)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1:4d}/{num_samples}] {base}  vf={volume_fraction:.3f}  "
                      f"C11={C_eff[0,0]:.4f}  k11={kappa_eff[0,0]:.4f}")

        except Exception as e:
            failed += 1
            print(f"  [{i+1:4d}/{num_samples}] FAILED: {base} — {e}")
            if failed > num_samples // 10:
                print("Too many failures, stopping.")
                break
            continue

    # ── Save metadata ─────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f" Done!  Successful: {len(records)}/{num_samples}  Failed: {failed}")
    print(f" Metadata → {json_path}")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMMStructures Dataset Sampler")
    parser.add_argument("--test",    action="store_true", help="Quick test with 5 samples")
    parser.add_argument("--n",       type=int, default=1000, help="Number of samples")
    parser.add_argument("--out-dir", type=str, default="Output/samples", help="Output directory")
    args = parser.parse_args()

    if args.test:
        print("[TEST MODE] Generating 5 samples...")
        build_dataset(num_samples=5, out_dir="Output/samples_test")
    else:
        build_dataset(num_samples=args.n, out_dir=args.out_dir)
