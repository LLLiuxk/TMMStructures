"""
Recompute Dataset via Multiprocessing
=====================================
Processes an existing dataset's parameters, loads the 1/4 cell generated images,
runs the (now corrected) homogenization code to compute full-cell C_eff,
and generates the radar plots using logarithmic normalization.
Multi-processing is used to drastically reduce compute time for large batches.
"""

import os
import json
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from homogenize import homogenize_elastic, homogenize_thermal, load_and_reconstruct
from plot_combined_radar import save_radar_chart

# Physics parameters
E0 = 1.0
nu = 0.3
k0 = 1.0
Emin = 1e-9
kmin = 1e-9
penal = 3.0

def process_single_record(args):
    """Worker function for multiprocessing."""
    i, record = args
    img_path = record.get("image_path", "")
    
    if not os.path.exists(img_path):
        return {"error": f"Image missing {img_path}", "index": i, "record": record}
            
    try:
        # Load and reconstruct (this now correctly applies 4-fold mirroring to 256x256)
        density = load_and_reconstruct(img_path, invert=False)
        nely, nelx = density.shape
        vf = float(np.mean(density))
        
        # Compute properties
        C_eff = homogenize_elastic(nelx, nely, density, E0, Emin, nu, penal)
        kappa_eff = homogenize_thermal(nelx, nely, density, k0, kmin, penal)
        
        # Verify if a standard radar path exists or generate one
        dataset_dir = os.path.dirname(os.path.dirname(img_path)) 
        radar_dir = os.path.join(dataset_dir, "radars")
        os.makedirs(radar_dir, exist_ok=True)
        out_radar = os.path.join(radar_dir, f"sample_{i:04d}_radar.png")
        
        # Use a neat title
        title = f"Sample {i:04d} (Symm Full Cell)\n$C_{{11}}/C_{{22}}$ = {C_eff[0,0]/max(C_eff[1,1], 1e-12):.1e}"
        # Regenerate the radar chart
        save_radar_chart(C_eff, kappa_eff, title, out_radar)
        
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
        record["radar_path"] = out_radar
        
        return {"result": record, "index": i}
        
    except Exception as e:
        record["properties"] = {"error": str(e)}
        return {"error": str(e), "index": i, "record": record}

def recompute_dataset(dataset_dir="Output/dataset/batch_1"):
    schema_path = os.path.join(dataset_dir, "dataset_schema.json")
    if not os.path.exists(schema_path):
        print(f"Error: dataset schema not found at {schema_path}")
        return
        
    with open(schema_path, "r") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records from {schema_path}")
    print(f"Using {cpu_count()} CPU cores for parallel processing...")
    
    start_time = time.time()
    
    # Pack arguments for multiprocessing
    task_args = [(i, record) for i, record in enumerate(records)]
    
    # To monitor progress, we'll use imap_unordered
    results_list = [None] * len(records)
    error_count = 0
    
    with Pool(processes=cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_record, task_args)):
            idx = res["index"]
            if "result" in res:
                results_list[idx] = res["result"]
            else:
                results_list[idx] = res["record"]
                print(f"\\nError in record {idx}: {res['error']}")
                error_count += 1
                
            # Log progress every 100 iterations or at the end
            completed = i + 1
            if completed % 100 == 0 or completed == len(records):
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(records) - completed) / rate if rate > 0 else 0
                print(f"\\rProgress: [{completed}/{len(records)}] - {completed/len(records)*100:.1f}% | "
                      f"{rate:.2f} it/s | ETA: {remaining/60:.1f}m", end="")
    
    print(f"\\n\\nProcessing complete! Time taken: {time.time() - start_time:.2f}s")
    print(f"Errors encountered: {error_count}")
    
    # Save processed dataset
    out_path = os.path.join(dataset_dir, "dataset_schema.json")
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Saved updated results to {out_path}")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "Output/dataset/batch_1"
    recompute_dataset(target_dir)
