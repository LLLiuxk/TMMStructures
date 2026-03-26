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

# CRITICAL: Prevent OpenMP thread explosion when using Multiprocessing with SciPy/NumPy 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from homogenize import homogenize_elastic, homogenize_thermal, load_and_reconstruct
from plot_combined_radar import save_radar_chart

# Physics parameters
E0 = 1.0
nu = 0.3
k0 = 1.0
Emin = 1e-9
kmin = 1e-9
penal = 3.0

def process_single_record(args_list):
    """Worker function for multiprocessing."""
    i, record, args = args_list
    img_path = record.get("image_path", "")
    
    if not os.path.exists(img_path):
        return {"error": f"Image missing {img_path}", "index": i, "record": record}
            
    try:
        # Load and reconstruct (this now correctly applies 4-fold mirroring to 256x256)
        # For batch_1 images, Black=Solid, so invert=True.
        density = load_and_reconstruct(img_path, invert=True)
        nely, nelx = density.shape
        vf = float(np.mean(density))
        
        # Compute properties
        C_eff = homogenize_elastic(nelx, nely, density, E0, Emin, nu, penal)
        kappa_eff = homogenize_thermal(nelx, nely, density, k0, kmin, penal)
        
        # Determine radar output directory
        if args.get("radar_dir"):
            radar_dir = args.get("radar_dir")
        else:
            # Default to the same directory as the source image
            radar_dir = os.path.dirname(img_path)
            
        os.makedirs(radar_dir, exist_ok=True)
        out_radar = os.path.join(radar_dir, f"sample_{i:04d}_radar.png")
        
        # Optionally regenerate the radar chart (this is slow, ~0.2s per image)
        if args.get("plot_radars", False):
            title = f"Sample {i:04d} (Symm Full Cell)\n$C_{{11}}/C_{{22}}$ = {C_eff[0,0]/max(C_eff[1,1], 1e-12):.1e}"
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

def recompute_dataset(dataset_dir="Output/dataset/batch_1", plot_radars=False, radar_dir=None, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    schema_path = os.path.join(dataset_dir, "dataset_schema.json")
    if not os.path.exists(schema_path):
        print(f"Error: dataset schema not found at {schema_path}")
        return
        
    with open(schema_path, "r") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records from {schema_path}")
    print(f"Using {cpu_count()} CPU cores for parallel processing...")
    print(f"Radar Plotting: {'ENABLED' if plot_radars else 'DISABLED (Fast Mode)'}")
    
    # Pack arguments for multiprocessing
    # Note: we pass whether to plot radars and the output directory
    task_args = []
    
    # Check for skipping existing records if requested
    skip_count = 0
    for i, record in enumerate(records):
        is_processed = "properties" in record and "C11" in record["properties"]
        has_radar = os.path.exists(record.get("radar_path", "")) if record.get("radar_path") else False
        
        # If skip_existing is True, we only add to task_args if not processed OR radar missing (if plotting enabled)
        if plot_radars:
            should_run = not (is_processed and has_radar)
        else:
            should_run = not is_processed
            
        if not should_run:
            skip_count += 1
            continue
            
        task_args.append((i, record, {"plot_radars": plot_radars, "radar_dir": radar_dir}))
    
    if skip_count > 0:
        print(f"Skipping {skip_count} already processed records.")
    
    if not task_args:
        print("All records already processed. Nothing to do.")
        return
        
    # To monitor progress, we'll use imap_unordered
    results_list = list(records) # Start with a copy to preserve existing data
    completed_count = 0
    total_to_process = len(task_args)
    error_count = 0
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_record, task_args)):
            idx = res["index"]
            if "result" in res:
                results_list[idx] = res["result"]
            else:
                results_list[idx] = res["record"]
                print(f"\nError in record {idx}: {res['error']}")
                error_count += 1
                
            # Log progress
            completed_count = i + 1
            if completed_count % 50 == 0 or completed_count == total_to_process:
                elapsed = time.time() - start_time
                rate = completed_count / elapsed
                remaining = (total_to_process - completed_count) / rate if rate > 0 else 0
                print(f"\rCurrent Progress: [{completed_count}/{total_to_process}] - {completed_count/total_to_process*100:.1f}% | "
                      f"{rate:.2f} it/s | ETA: {remaining/60:.1f} min  ", end="", flush=True)
                      
            # Auto-save every 1000 iterations to prevent data loss
            if completed_count % 1000 == 0:
                with open(schema_path, "w") as f:
                    json.dump(results_list, f, indent=2)
    
    print(f"\n\nProcessing complete! Time taken: {time.time() - start_time:.2f}s")
    print(f"Errors encountered: {error_count}")
    
    # Save processed dataset
    out_path = os.path.join(dataset_dir, "dataset_schema.json")
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Saved updated results to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", nargs="?", default="Output/dataset/batch_1", help="Target directory (containing dataset_schema.json)")
    parser.add_argument("-r", "--radar", action="store_true", help="Enable radar plot regeneration (SLOW)")
    parser.add_argument("--radar-dir", help="Custom directory for radar plots (defaults to image folder)")
    parser.add_argument("-n", "--num-workers", type=int, default=max(1, cpu_count() - 1), help="Number of workers (default: CPU-1)")
    args = parser.parse_args()
    
    recompute_dataset(args.target_dir, plot_radars=args.radar, radar_dir=args.radar_dir, num_workers=args.num_workers)
