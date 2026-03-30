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
import itertools
from generate_microstructure import render_microstructure
from homogenize import process_image
from plot_combined_radar import save_radar_chart

EDGES = ["E1", "E2", "E3", "E4"]


class CompactJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that keeps short lists (e.g. matrix rows) on a single line."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent_level = 0

    def encode(self, o):
        return self._encode(o, self._indent_level)

    def _encode(self, o, indent_level):
        indent_str = "  " * indent_level
        child_indent = "  " * (indent_level + 1)

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k, v in o.items():
                val = self._encode(v, indent_level + 1)
                items.append(f"{child_indent}{json.dumps(k)}: {val}")
            return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"

        if isinstance(o, list):
            if not o:
                return "[]"
            # If all elements are simple scalars (numbers, strings, bools, None), put on one line
            if all(isinstance(x, (int, float, bool, type(None), str)) for x in o):
                return "[" + ", ".join(json.dumps(x) for x in o) + "]"
            # If this is a list-of-lists where inner lists are all simple (e.g. matrix), one row per line
            if all(isinstance(x, list) and all(isinstance(y, (int, float, bool, type(None))) for y in x) for x in o):
                rows = ["[" + ", ".join(json.dumps(y) for y in row) + "]" for row in o]
                return "[\n" + ",\n".join(child_indent + r for r in rows) + "\n" + indent_str + "]"
            # Otherwise use multi-line format
            items = []
            for item in o:
                items.append(child_indent + self._encode(item, indent_level + 1))
            return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"

        return json.dumps(o)


def dump_compact_json(data, f):
    """Write data to file using compact JSON formatting."""
    f.write(CompactJSONEncoder().encode(data))

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


def sample_adjacency(n, config, max_attempts=50000, rng=random):
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
        p = rng.uniform(sp_min, sp_max)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
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

def enumerate_all_topologies(n, config):
    """
    Exhaustively enumerates all possible valid symmetric adjacency matrices for n nodes.
    Filters by connectivity and max degree constraints.
    """
    max_degree = config.get("max_node_degree", 2)
    # Get all possible undirected edges (i, j) where i < j
    possible_edges = list(itertools.combinations(range(n), 2))
    num_possible = len(possible_edges)
    
    valid_adjs = []
    # Iterate through all 2^num_possible combinations
    # For n=4, 2^6=64; n=5, 2^10=1024; n=6, 2^15=32768 (still feasible)
    for combo in itertools.product([0, 1], repeat=num_possible):
        adj = [[0] * n for _ in range(n)]
        for is_present, (u, v) in zip(combo, possible_edges):
            if is_present:
                adj[u][v] = adj[v][u] = 1
        
        if is_connected_and_valid(adj, n, max_degree):
            valid_adjs.append(adj)
    return valid_adjs

def generate_grid_schemas(config):
    """
    Pure deterministic generator. Performs a full Cartesian product over:
    1. Node count combinations (npe_combo)
    2. Valid topological skeletons (adjacency matrices)
    3. Connection types for each edge
    4. Node positions and widths
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
    
    # 1. Iterate Node Counts
    for npe_combo in itertools.product(npe_pool, repeat=4):
        total_nodes = sum(npe_combo)
        if total_nodes == 0: continue
            
        # 2. Iterate ALL Valid Topologies
        all_adjs = enumerate_all_topologies(total_nodes, config)
        
        # Prepare node index mapping to edges
        node_edge_list = []
        for edge_idx, edge_name in enumerate(EDGES):
            for _ in range(npe_combo[edge_idx]):
                node_edge_list.append(edge_name)

        for adj in all_adjs:
            # Identify active connections (edges) in the current adjacency matrix
            active_edges = []
            for i in range(total_nodes):
                for j in range(i + 1, total_nodes):
                    if adj[i][j]:
                        active_edges.append((i, j))
            
            num_conns = len(active_edges)

            # 3. Iterate ALL combinations of connection types for these edges
            for type_combo in itertools.product(allowed_types, repeat=num_conns):
                
                # 4. Iterate ALL Position combinations
                for pos_combo in itertools.product(pos_pool, repeat=total_nodes):
                    
                    # 5. Iterate ALL Width combinations
                    for width_combo in itertools.product(width_pool, repeat=total_nodes):
                        
                        nodes_dict = {e: [] for e in EDGES}
                        # Temp storage to resolve indices in connections
                        node_info = [] # list of (edge, pos, width)
                        
                        for i in range(total_nodes):
                            edge_name = node_edge_list[i]
                            p_val = pos_combo[i]
                            w_val = width_combo[i]
                            nodes_dict[edge_name].append([p_val, w_val])
                            node_info.append((edge_name, p_val, w_val))

                        # Build connections
                        connections = []
                        for conn_idx, (u, v) in enumerate(active_edges):
                            # Find index in the specific edge list for rendering
                            u_edge, u_p, u_w = node_info[u]
                            v_edge, v_p, v_w = node_info[v]
                            
                            u_idx = nodes_dict[u_edge].index([u_p, u_w])
                            v_idx = nodes_dict[v_edge].index([v_p, v_w])
                            
                            connections.append({
                                "start": [u_edge, u_idx],
                                "end":   [v_edge, v_idx],
                                "type":  type_combo[conn_idx]
                            })
                        
                        yield {
                            "nodes": nodes_dict,
                            "node_list": [[e, i] for e, i in zip(node_edge_list, range(total_nodes))],
                            "matrix": adj,
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

def build_dataset(config, offset=0, fast_mode=False, resume=True):
    out_dir = config.get("output_dir", "Output/dataset")
    num_samples = config.get("num_samples", 10)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    
    db_path = os.path.join(out_dir, "dataset_schema.json")
    dataset_records = []
    
    # Checkpoint / Resume logic
    existing_ids = set()
    if resume and os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                dataset_records = json.load(f)
            print(f"Loaded existing database with {len(dataset_records)} records from {db_path}")
            # Collect already generated IDs
            existing_ids = {rec["id"] for rec in dataset_records}
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse existing {db_path}. Starting fresh.")
    
    failed = 0
    skipped = 0
    
    print(f"{'=' * 60}")
    print(f" TMMStructures - Generating {num_samples} microstructures")
    print(f" Output → {os.path.abspath(out_dir)}")
    print(f" Config: nodes_per_edge={config.get('nodes_per_edge_range', [1,1])}, "
          f"max_degree={config.get('max_node_degree', 2)}")
    print(f" Offset: {offset}  (sample IDs: {offset} ~ {offset + num_samples - 1})")
    print(f" Compute: {'SKIP (Fast Mode)' if fast_mode else 'ON (FEM & Radar)'}")
    print(f" Resume:  {'ON' if resume else 'OFF'}")
    print(f"{'=' * 60}")
    
    base_seed = config.get("random_seed", None)
    if base_seed is not None:
        random.seed(base_seed + offset)
        np.random.seed(base_seed + offset)
        print(f" Global random seed set to: {base_seed} + offset {offset} = {base_seed + offset}")

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
            i = 0
            while True:
                # For random mode, advance the seed by (offset + i) to ensure
                # different offsets produce entirely different sample sequences
                seed_val = base_seed + offset + i if base_seed is not None else None
                yield generate_random_schema(config, seed=seed_val)
                i += 1
        schema_generator = random_schema_generator()
    
    for i in range(num_samples):
        actual_id = offset + i
        base_name = f"sample_{actual_id:05d}"
        
        # Check if we should skip: BOTH json record AND image file must exist
        img_path_check = os.path.join(out_dir, "images", f"{base_name}.png")
        if resume and base_name in existing_ids and os.path.exists(img_path_check):
            skipped += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1:4d}/{num_samples}] {base_name}  ...skipped (already exists)")
            
            # We STILL MUST advance the generator to keep sequence aligned!
            try:
                next(schema_generator)
            except StopIteration:
                pass
            continue
            
        img_path = os.path.join(out_dir, "images", f"{base_name}.png")
        radar_path = os.path.join(out_dir, "images", f"{base_name}_radar.png")
        
        try:
            # Get next schema from our generator
            schema = next(schema_generator)
            
            # 1. Render Image
            render_schema = {
                "nodes": schema["nodes"],
                "connections": schema["connections"]
            }
            render_microstructure(render_schema, size=(128, 128), out_path=img_path)
            
            # 2. & 3. Homogenize and Radar Plot (Conditional)
            props = {}
            if not fast_mode:
                # Calculate Mechanical and Thermal properties
                props = process_image(img_path, silent=True)
                
                # Generate Radar Plot
                save_radar_chart(
                    np.array(props['C_eff']), 
                    np.array(props['kappa_eff']), 
                    f"Properties: {base_name}", 
                    radar_path
                )
            
            record = {
                "id": base_name,
                "image_path": img_path,
                "radar_path": radar_path if not fast_mode else "",
                "schema": schema,
                "properties": props
            }
            dataset_records.append(record)
            
            if (i + 1) % 50 == 0 or i == 0:
                n_nodes = len(schema["node_list"])
                if fast_mode:
                    print(f"  [{i+1:4d}/{num_samples}] {base_name}  "
                          f"nodes={n_nodes}  [Fast Mode: Compute Skipped]")
                else:
                    vf = props.get('volume_fraction', 0.0)
                    c11 = props.get('C11', 0.0)
                    print(f"  [{i+1:4d}/{num_samples}] {base_name}  "
                          f"nodes={n_nodes}  vf={vf:.3f}  C11={c11:.3e}")
                
            # Periodically save the database to prevent data loss
            if (i + 1) % 100 == 0:
                with open(db_path, "w") as f:
                    dump_compact_json(dataset_records, f)
                
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
    
    # Final save
    with open(db_path, "w") as f:
        dump_compact_json(dataset_records, f)
        
    try:
        from plot_property_coverage import generate_coverage_plot
        print("\nGenerating property coverage summary plot...")
        generate_coverage_plot(out_dir, show_plot=False)
    except Exception as e:
        print(f"Warning: Failed to generate property coverage plot: {e}")
    
    print(f"\n{'=' * 60}")
    print(f" Done!  Successful: {len(dataset_records) - skipped}/{num_samples}  Skipped: {skipped}  Failed: {failed}")
    print(f" Total records in database: {len(dataset_records)}")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="dataset_config.json", help="Path to config file")
    parser.add_argument("--offset", type=int, default=0, help="Starting index ID for generated samples (useful for split-PC generation)")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples to generate")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip FEM properties and radar charts (images & schema only)")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpointing and start fresh (overwrite existing records)")
    args = parser.parse_args()

    # Apply configuration and overrides
    cfg = get_or_create_config(args.config)
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
        
    build_dataset(cfg, offset=args.offset, fast_mode=args.fast, resume=not args.no_resume)
