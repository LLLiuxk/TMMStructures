"""
Microstructure Dataset Generator (1 Point Per Edge, Variable Width at Nodes)
==========================================================================
Randomly samples parameters within bounded rules to generate a batch of 
1/4 symmetry unit cell structures. Renders them to images and saves
their governing parameters.

FEATURES:
1. Exactly 1 node per outer edge (E1, E2, E3, E4) -> Total 4 nodes.
2. Nodes contain both [position, width].
3. Connection topology is directly sampled via a 4x4 adjacency matrix.
4. Connections are just defined by logical links (no built-in type or width).
5. MUST form a single connected component for the 4 nodes.
"""

import os
import json
import numpy as np
import random
from generate_microstructure import render_microstructure

def is_connected_and_valid(adj_matrix):
    """
    Checks if a 4x4 adjacency matrix forms a single connected graph
    and each node has a maximum degree of 2.
    """
    n = 4
    
    # Check degree constraint
    for i in range(n):
        if sum(adj_matrix[i]) > 2:
            return False
            
    # Check connectivity via BFS
    visited = set()
    queue = [0] # start at node 0
    visited.add(0)
    
    while queue:
        curr = queue.pop(0)
        for neighbor in range(n):
            if adj_matrix[curr][neighbor] == 1 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return len(visited) == n

def generate_random_schema(seed=None):
    """
    Generates a random valid 1/4 microstructure schema dictionary.
    Samples a 4x4 matrix for topology.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1 point per edge, each node comes with [pos, width]
    nodes = {
        "E1": [[round(random.uniform(0.1, 0.9), 3), round(random.uniform(0.05, 0.20), 3)]],
        "E2": [[round(random.uniform(0.1, 0.9), 3), round(random.uniform(0.05, 0.20), 3)]],
        "E3": [[round(random.uniform(0.1, 0.9), 3), round(random.uniform(0.05, 0.20), 3)]],
        "E4": [[round(random.uniform(0.1, 0.9), 3), round(random.uniform(0.05, 0.20), 3)]]
    }

    # Map indices 0-3 to specific node references
    node_mapping = [
        ("E1", 0),
        ("E2", 0),
        ("E3", 0),
        ("E4", 0)
    ]

    max_attempts = 10000
    for attempt in range(max_attempts):
        # Build a random 4x4 symmetric adjacency matrix
        adj_matrix = np.zeros((4, 4), dtype=int)
        
        # Decide sparsity
        p_edge = random.uniform(0.3, 0.8)
        
        for i in range(4):
            for j in range(i + 1, 4):
                if random.random() < p_edge:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
                    
        # Check connectivity and max degree constraints
        if not is_connected_and_valid(adj_matrix):
            continue 
            
        # Matrix is valid! Build connections list.
        connections = []
        for i in range(4):
            for j in range(i + 1, 4):
                if adj_matrix[i][j] == 1:
                    c_type = random.choice(["straight_line", "bezier_curve", "circular_arc", "tapered_line"])
                    conn = {
                        "start": list(node_mapping[i]),
                        "end": list(node_mapping[j]),
                        "type": c_type
                    }
                    connections.append(conn)
                    
        return {
            "nodes": nodes,
            "matrix": adj_matrix.tolist(),
            "connections": connections
        }

    raise Exception("Could not find a valid single-connected matrix within attempt limit.")

def build_dataset(num_samples=10, out_dir="Output/dataset"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    
    dataset_records = []
    
    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        schema = generate_random_schema(seed=42 + i)
        
        base_name = f"sample_{i:04d}"
        img_path = os.path.join(out_dir, "images", f"{base_name}.png")
        
        try:
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
        except Exception as e:
            print(f"Failed to render {base_name}: {e}")
            continue
            
    db_path = os.path.join(out_dir, "dataset_schema.json")
    with open(db_path, "w") as f:
        json.dump(dataset_records, f, indent=2)
        
    print(f"Dataset generated at {out_dir}")
    print(f"Total samples: {len(dataset_records)}")

if __name__ == "__main__":
    # Test run generating 5 samples for review
    build_dataset(5)
