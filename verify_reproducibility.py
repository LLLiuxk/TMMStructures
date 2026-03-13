import json
import os
import shutil
import subprocess

def run_gen(mode, seed, num_samples=3):
    config = {
        "num_samples": num_samples,
        "random_seed": seed,
        "sampling_mode": mode,
        "output_dir": f"Test/output_{mode}_{seed}",
        "node_position_range": [0.1, 0.9],
        "node_position_step": 0.4,
        "node_width_range": [0.1, 0.2],
        "node_width_step": 0.1,
        "nodes_per_edge_range": [1, 1],
        "max_node_degree": 2,
        "sparsity_range": [0.5, 0.5],
        "allowed_connection_types": ["bezier_curve"]
    }
    with open("test_config.json", "w") as f:
        json.dump(config, f)
    
    # Run generate_dataset.py with the test config
    # We'll modify generate_dataset.py slightly to accept a config path or just overwrite dataset_config.json
    shutil.copy("test_config.json", "dataset_config.json")
    subprocess.run(["python", "generate_dataset.py"], capture_output=True)
    
    schema_path = os.path.join(config["output_dir"], "dataset_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return json.load(f)
    return None

def main():
    os.makedirs("Test", exist_ok=True)
    
    print("Testing RANDOM mode reproducibility...")
    r1 = run_gen("random", 42)
    r2 = run_gen("random", 42)
    
    if r1 == r2 and r1 is not None:
        print("✅ RANDOM mode is reproducible.")
    else:
        print("❌ RANDOM mode is NOT reproducible.")
        # print(json.dumps(r1, indent=2))
        # print(json.dumps(r2, indent=2))

    print("\nTesting GRID mode reproducibility...")
    g1 = run_gen("grid", 42)
    g2 = run_gen("grid", 42)
    
    if g1 == g2 and g1 is not None:
        print("✅ GRID mode is reproducible.")
    else:
        print("❌ GRID mode is NOT reproducible.")
        if g1 and g2:
             # Compare first sample's matrix
             m1 = g1[0]['schema']['matrix']
             m2 = g2[0]['schema']['matrix']
             print(f"Matrix 1: {m1}")
             print(f"Matrix 2: {m2}")

if __name__ == "__main__":
    main()
