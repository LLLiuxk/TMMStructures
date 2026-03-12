"""
Microstructure Dataset Generator (Config-Driven, Variable Nodes Per Edge)
=========================================================================
Reads all generation parameters from dataset_config.json and produces
a batch of 1/4 symmetry unit cell microstructure images.

Configurable parameters:
  - num_samples:              Number of images to generate
  - output_dir:               Output directory
  - node_position_range:      [min, max] for node position t on each edge
  - node_width_range:         [min, max] for node width d
  - nodes_per_edge_range:     [min, max] for number of nodes placed on each edge
  - max_node_degree:          Maximum connection degree allowed for any single node
  - sparsity_range:           [min, max] probability range for edge existence
  - allowed_connection_types: List of connection type strings
"""

import os
import json
import random
import numpy as np
from generate_microstructure import render_microstructure

EDGES = ["E1", "E2", "E3", "E4"]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Topology Generation
# ═══════════════════════════════════════════════════════════════════════════════

def sample_nodes_on_edge(n_nodes, pos_min, pos_max):
    """
    Sample n_nodes positions on a single edge within [pos_min, pos_max].
    If n_nodes >= 2, ensure any two positions are at least 0.15 apart.
    Returns a sorted list of positions.
    """
    if n_nodes == 1:
        return [round(random.uniform(pos_min, pos_max), 4)]
    
    for _ in range(500):
        positions = [random.uniform(pos_min, pos_max) for _ in range(n_nodes)]
        positions.sort()
        # Check minimum separation between adjacent positions
        valid = True
        for i in range(len(positions) - 1):
            if positions[i + 1] - positions[i] < 0.15:
                valid = False
                break
        if valid:
            return [round(p, 4) for p in positions]
    
    # Fallback: evenly space within range
    step = (pos_max - pos_min) / (n_nodes + 1)
    return [round(pos_min + step * (i + 1), 4) for i in range(n_nodes)]


def build_nodes_config(config):
    """
    Randomly decide how many nodes to place on each edge based on config,
    then sample their positions and widths.
    Returns:
        nodes_dict:  dict  {edge_id: [[pos, width], ...]}  (render format)
        node_list:   list  [(edge_id, node_idx), ...]      (for matrix indexing)
    """
    npe_min, npe_max = config.get("nodes_per_edge_range", [1, 1])
    pos_min, pos_max = config.get("node_position_range", [0.1, 0.9])
    width_min, width_max = config.get("node_width_range", [0.05, 0.20])
    
    nodes_dict = {}
    node_list = []
    
    for edge in EDGES:
        n = random.randint(npe_min, npe_max)
        positions = sample_nodes_on_edge(n, pos_min, pos_max)
        
        edge_nodes = []
        for pos in positions:
            w = round(random.uniform(width_min, width_max), 4)
            # Clamp width so that [pos - w/2, pos + w/2] stays within (0, 1)
            max_w = 2 * min(pos - 0.01, 0.99 - pos)
            w = min(w, max_w)
            w = max(w, 0.01)
            edge_nodes.append([pos, round(w, 4)])
        
        nodes_dict[edge] = edge_nodes
        for idx in range(n):
            node_list.append((edge, idx))
    
    return nodes_dict, node_list


def is_connected_and_valid(adj, n, max_degree):
    """
    Checks if an n×n adjacency matrix forms a single connected graph
    and each node has degree in [1, max_degree].
    """
    # Check degree constraints
    for i in range(n):
        deg = sum(adj[i])
        if deg < 1 or deg > max_degree:
            return False
    
    # Check connectivity via BFS
    visited = {0}
    queue = [0]
    while queue:
        curr = queue.pop(0)
        for nb in range(n):
            if adj[curr][nb] and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


def sample_adjacency(n, config, max_attempts=50000):
    """
    Sample a random valid n×n symmetric adjacency matrix satisfying:
      - degree in [1, max_degree] for every node
      - single connected component
    Returns adj (list of lists), or None on failure.
    """
    max_degree = config.get("max_node_degree", 2)
    sp_min, sp_max = config.get("sparsity_range", [0.3, 0.8])
    
    for _ in range(max_attempts):
        adj = [[0] * n for _ in range(n)]
        p = random.uniform(sp_min, sp_max)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        if is_connected_and_valid(adj, n, max_degree):
            return adj
    return None


import itertools

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Schema Generation
# ═══════════════════════════════════════════════════════════════════════════════

def get_grid_range(min_val, max_val, step):
    """Returns a list of values from min_val to max_val inclusive, separated by step."""
    if step <= 0: return [min_val]
    vals = []
    curr = min_val
    while curr <= max_val + 1e-9:
        vals.append(round(curr, 4))
        curr += step
    return vals

def generate_grid_schemas(config):
    """
    Generator that evaluates deterministic gridding properties and yields
    every combination of grid position and grid width combinations via Cartesian product.
    Iterates over the nodes_per_edge range as well.
    Fixes a specific random topology graph for each npe count first.
    """
    npe_min, npe_max = config.get("nodes_per_edge_range", [1, 1])
    
    pos_min, pos_max = config.get("node_position_range", [0.1, 0.9])
    pos_step = config.get("node_position_step", 0.4)
    pos_pool = get_grid_range(pos_min, pos_max, pos_step)
    
    width_min, width_max = config.get("node_width_range", [0.05, 0.20])
    width_step = config.get("node_width_step", 0.05)
    width_pool = get_grid_range(width_min, width_max, width_step)
    
    allowed_types = config.get("allowed_connection_types", ["straight_line"])
    if not allowed_types: allowed_types = ["straight_line"]

    npe_pool = range(npe_min, npe_max + 1)
    
    # Iterate over all possible combinations of node counts for the 4 edges
    for npe_combo in itertools.product(npe_pool, repeat=4):
        total_nodes = sum(npe_combo)
        
        # Guard against zero total nodes
        if total_nodes == 0:
            continue
            
        # Generate one valid topology to map parameters onto for this npe combination
        base_adj = None
        for _ in range(5000):
            adj = sample_adjacency(total_nodes, config)
            if adj is not None:
                base_adj = adj
                break
                
        if base_adj is None:
            print(f"Warning: Failed to generate base adjacency for grid search with npe_combo={npe_combo}")
            continue

        # Generate edge-by-edge node index lists
        node_list = []
        for edge_idx, edge in enumerate(EDGES):
            for _ in range(npe_combo[edge_idx]):
                node_list.append(edge)
                
        # Iterate through all combinations of positions and widths for ALL nodes
        # For n nodes, we need n pos values and n width values
        for pos_combo in itertools.product(pos_pool, repeat=total_nodes):
            for width_combo in itertools.product(width_pool, repeat=total_nodes):
                
                nodes_dict = {edge: [] for edge in EDGES}
                
                # Populate node dict
                for i in range(total_nodes):
                    edge = node_list[i]
                    nodes_dict[edge].append([pos_combo[i], width_combo[i]])
                    
                # Build connections
                connections = []
                for i in range(total_nodes):
                    for j in range(i + 1, total_nodes):
                        if base_adj[i][j]:
                            connections.append({
                                "start": [node_list[i], nodes_dict[node_list[i]].index([pos_combo[i], width_combo[i]])],
                                "end":   [node_list[j], nodes_dict[node_list[j]].index([pos_combo[j], width_combo[j]])],
                                "type":  random.choice(allowed_types)  # Alternatively, grid over connection types too
                            })
                
                yield {
                    "nodes": nodes_dict,
                    "node_list": [[e, i] for e, i in zip(node_list, range(total_nodes))],
                    "matrix": base_adj,
                    "connections": connections
                }

def generate_random_schema(config, seed=None):
    """
    Generates a random valid microstructure schema dictionary.
    Supports variable nodes per edge and configurable degree constraints.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    for attempt in range(200):
        nodes_dict, node_list = build_nodes_config(config)
        n = len(node_list)
        
        adj = sample_adjacency(n, config)
        if adj is None:
            continue  # Retry with new node layout
        
        # Build connections
        allowed_types = config.get("allowed_connection_types",
                                   ["straight_line", "bezier_curve", "circular_arc", "tapered_line"])
        if not allowed_types:
            allowed_types = ["straight_line"]
        
        connections = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    c_type = random.choice(allowed_types)
                    connections.append({
                        "start": list(node_list[i]),
                        "end":   list(node_list[j]),
                        "type":  c_type
                    })
        
        return {
            "nodes": nodes_dict,
            "node_list": [[e, idx] for (e, idx) in node_list],
            "matrix": adj,
            "connections": connections
        }
    
    raise RuntimeError("Could not generate a valid schema within attempt limit.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(config):
    out_dir = config.get("output_dir", "Output/dataset")
    num_samples = config.get("num_samples", 10)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    
    dataset_records = []
    failed = 0
    
    print(f"{'=' * 60}")
    print(f" TMMStructures - Generating {num_samples} microstructures")
    print(f" Output → {os.path.abspath(out_dir)}")
    print(f" Config: nodes_per_edge={config.get('nodes_per_edge_range', [1,1])}, "
          f"max_degree={config.get('max_node_degree', 2)}")
    print(f"{'=' * 60}")
    
    base_seed = config.get("random_seed", None)
    sampling_mode = config.get("sampling_mode", "random")
    
    # Set up generator based on mode
    if sampling_mode == "grid":
        print(" Using GridSearchCV generator (num_samples will limit total output).")
        try:
            schema_generator = generate_grid_schemas(config)
        except Exception as e:
            print(f"Failed to initialize grid generator: {e}")
            return
    else:
        # Dummy generator wrapper for random mode
        def random_schema_generator():
            while True:
                yield generate_random_schema(config, seed=base_seed)
        schema_generator = random_schema_generator()
    
    for i in range(num_samples):
        base_name = f"sample_{i:04d}"
        img_path = os.path.join(out_dir, "images", f"{base_name}.png")
        
        try:
            if sampling_mode == "random":
                # For random mode, we optionally advance the seed each time
                seed_val = base_seed + i if base_seed is not None else None
                schema = generate_random_schema(config, seed=seed_val)
            else:
                # For grid mode, just take the next from generator
                schema = next(schema_generator)
            
            render_schema = {
                "nodes": schema["nodes"],
                "connections": schema["connections"]
            }
            render_microstructure(render_schema, size=(128, 128), out_path=img_path)
            
            record = {
                "id": base_name,
                "image_path": img_path,
                "schema": schema,
            }
            dataset_records.append(record)
            
            if (i + 1) % 50 == 0 or i == 0:
                n_nodes = len(schema["node_list"])
                n_conns = len(schema["connections"])
                print(f"  [{i+1:4d}/{num_samples}] {base_name}  "
                      f"nodes={n_nodes}  connections={n_conns}")
                
        except StopIteration:
            print(f"\nGrid search exhausted all permutations after {i} samples.")
            break
        except Exception as e:
            failed += 1
            print(f"  [{i+1:4d}/{num_samples}] FAILED: {base_name} — {e}")
            if failed > num_samples // 10 and failed > 5:
                print("Too many failures, stopping early.")
                break
            continue
    
    db_path = os.path.join(out_dir, "dataset_schema.json")
    with open(db_path, "w") as f:
        json.dump(dataset_records, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f" Done!  Successful: {len(dataset_records)}/{num_samples}  Failed: {failed}")
    print(f" Metadata → {db_path}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Configuration File Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_or_create_config(config_path="dataset_config.json"):
    default_config = {
        "num_samples": 10,
        "output_dir": "Output/dataset",
        "random_seed": None,
        "node_position_range": [0.1, 0.9],
        "node_width_range": [0.05, 0.20],
        "nodes_per_edge_range": [1, 2],
        "max_node_degree": 2,
        "sparsity_range": [0.3, 0.8],
        "allowed_connection_types": [
            "straight_line",
            "bezier_curve",
            "circular_arc",
            "tapered_line"
        ]
    }
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found. Creating default at {config_path}")
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    
    print(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    config = get_or_create_config()
    build_dataset(config)
