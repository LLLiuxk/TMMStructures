import os
import json
import csv
import glob
import numpy as np
from homogenize import homogenize_elastic, homogenize_thermal, load_and_reconstruct
from plot_combined_radar import save_radar_chart

# Constants (Material properties used in Batch 1)
E0 = 1.0
nu = 0.3
k0 = 1.0
Emin = 1e-9
kmin = 1e-9
penal = 3.0

def process_folder(test_dir):
    """
    Scans the test_dir for *_topo.jpg files, calculates properties, and plots radar charts.
    """
    image_files = sorted(glob.glob(os.path.join(test_dir, "*_topo.jpg")))
    
    if not image_files:
        print(f"No *_topo.jpg files found in {test_dir}")
        return

    print(f"Processing {len(image_files)} microstructures in {test_dir}...")
    
    results = []
    
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(".jpg", "")
        # The user's files are 100x100 quarter cells. 
        # invert=True because Black=Solid, White=Void.
        try:
            density = load_and_reconstruct(img_path, invert=True)
            nely, nelx = density.shape
            vf = float(np.mean(density))
            
            # 1. Calculate properties
            C_eff = homogenize_elastic(nelx, nely, density, E0, Emin, nu, penal)
            kappa_eff = homogenize_thermal(nelx, nely, density, k0, kmin, penal)
            
            # 2. Generate Radar Plot
            # Using log-scale title format similar to recompute script
            title = f"{base_name} (200x200 Full Cell)\nVF: {vf:.3f} | $C_{{11}}/C_{{22}}$: {C_eff[0,0]/max(C_eff[1,1], 1e-12):.1e}"
            radar_path = os.path.join(test_dir, f"{base_name}_radar.png")
            save_radar_chart(C_eff, kappa_eff, title, radar_path)
            
            # 3. Store results
            record = {
                "id": base_name,
                "image_path": img_path,
                "radar_path": radar_path,
                "properties": {
                    "volume_fraction": vf,
                    "C_eff": C_eff.tolist(),
                    "kappa_eff": kappa_eff.tolist(),
                    "C11": float(C_eff[0, 0]),
                    "C12": float(C_eff[0, 1]),
                    "C22": float(C_eff[1, 1]),
                    "C66": float(C_eff[2, 2]),
                    "C16": float(C_eff[0, 2]),
                    "C26": float(C_eff[1, 2]),
                    "k11": float(kappa_eff[0, 0]),
                    "k12": float(kappa_eff[0, 1]),
                    "k22": float(kappa_eff[1, 1]),
                }
            }
            results.append(record)
            print(f"  Processed {base_name}: C11={C_eff[0,0]:.3e}, k11={kappa_eff[0,0]:.3e}")
            
        except Exception as e:
            print(f"  Error processing {base_name}: {e}")

    # Output JSON
    json_path = os.path.join(test_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Output CSV for easy reading
    csv_path = os.path.join(test_dir, "results.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "VF", "C11", "C22", "C12", "C66", "k11", "k22", "k12"])
        for r in results:
            p = r["properties"]
            writer.writerow([
                r["id"], f"{p['volume_fraction']:.4f}",
                f"{p['C11']:.4e}", f"{p['C22']:.4e}", f"{p['C12']:.4e}", f"{p['C66']:.4e}",
                f"{p['k11']:.4e}", f"{p['k22']:.4e}", f"{p['k12']:.4e}"
            ])
            
    print(f"\nProcessing complete! Results saved to {test_dir}")

if __name__ == "__main__":
    target_folder = os.path.join("output", "dataset", "test")
    process_folder(target_folder)
