"""
Process Dataset
===============
Runs the thermo-mechanical homogenization (homogenize.py) 
on a generated parameter dataset.
Combines the generating schemas with the computed C_eff and kappa_eff.
"""

import os
import json
import numpy as np
import time
from homogenize import homogenize_elastic, homogenize_thermal, load_and_reconstruct

def process_dataset_batch(dataset_dir="Output/dataset"):
    schema_path = os.path.join(dataset_dir, "dataset_schema.json")
    if not os.path.exists(schema_path):
        print(f"Error: dataset schema not found at {schema_path}")
        return
        
    with open(schema_path, "r") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records from {schema_path}")
    
    # Physics parameters
    E0 = 1.0
    nu = 0.3
    k0 = 1.0
    Emin = 1e-9
    kmin = 1e-9
    penal = 3.0
    
    start_time = time.time()
    
    results = []
    
    for i, record in enumerate(records):
        img_path = record["image_path"]
        if not os.path.exists(img_path):
            print(f"Warning: Image missing {img_path}")
            continue
            
        print(f"Processing [{i+1}/{len(records)}]: {img_path}")
        
        # We generate black=void, white=solid in our generator
        # So invert=False is needed here since homogenize default is invert=True
        # Let's check how we generate them: Image.new('L', size, color=0) is black.
        # We draw lines with fill=255 (white). So white=solid.
        density = load_and_reconstruct(img_path, invert=False)
        
        nely, nelx = density.shape
        vf = float(np.mean(density))
        
        # Compute properties
        try:
            C_eff = homogenize_elastic(nelx, nely, density, E0, Emin, nu, penal)
            kappa_eff = homogenize_thermal(nelx, nely, density, k0, kmin, penal)
            
            # Pack results back into record
            record["properties"] = {
                "volume_fraction": vf,
                "C_eff": C_eff.tolist(),
                "kappa_eff": kappa_eff.tolist(),
                "C11": C_eff[0, 0],
                "C12": C_eff[0, 1],
                "C22": C_eff[1, 1],
                "C66": C_eff[2, 2],
                "k11": kappa_eff[0, 0],
                "k22": kappa_eff[1, 1]
            }
            results.append(record)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            record["properties"] = {"error": str(e)}
            results.append(record)
            
    # Save processed DB
    out_path = os.path.join(dataset_dir, "dataset_with_properties.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Time taken: {elapsed:.2f}s")
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    process_dataset_batch()
