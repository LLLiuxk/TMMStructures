import json
import os
import numpy as np
import time
from homogenize import homogenize_elastic, load_and_reconstruct
import matplotlib.pyplot as plt

def preview_serial():
    dataset_dir = "Output/dataset/batch_1"
    schema_path = os.path.join(dataset_dir, "dataset_schema.json")
    with open(schema_path, "r") as f:
        records = json.load(f)[:50] # 50 samples only for speed and stability
        
    print(f"Generating SERIAL preview for 50 records...")
    vfs, c11s, ks = [], [], []
    start = time.time()
    for i, record in enumerate(records):
        img_path = record.get("image_path", "")
        density = load_and_reconstruct(img_path, invert=True)
        C_eff = homogenize_elastic(256, 256, density, 1.0, 1e-9, 0.3, 3.0)
        vfs.append(np.mean(density))
        c11s.append(C_eff[0,0])
        ks.append((C_eff[0,0] + C_eff[1,1] + 2*C_eff[0,1]) / 4.0)
        print(f"  {i+1}/50 done (current C11: {C_eff[0,0]:.2e})")

    print(f"Computed 50 samples in {time.time()-start:.1f}s")
    
    # Save a small JSON for the plot comparison
    preview_data = {"vfs": vfs, "c11s": c11s, "ks": ks}
    with open("preview_data.json", "w") as f:
        json.dump(preview_data, f)

    vf_plot = np.linspace(0.001, 1, 100)
    K0 = 1.0 / (2 * (1 - 0.3)); G0 = 1.0 / (2 * (1 + 0.3))
    K_hs = [K0 + (1-f) / (1/(0 - K0) + f/(K0 + G0)) for f in vf_plot]
    
    plt.figure(figsize=(8, 6))
    plt.plot(vf_plot, K_hs, 'r--', label='HS Upper Bound (Bulk Modulus)')
    plt.scatter(vfs, ks, s=30, alpha=0.7, label='Corrected Batch 1 Samples (50 pts)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(0.05, 1); plt.ylim(1e-4, 2)
    plt.title("PREVIEW: Recovered Property Cloud (Log-Log)", fontsize=14, weight='bold')
    plt.xlabel('Volume Fraction $\phi$'); plt.ylabel('Bulk Modulus $K$')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    out_path = "C:\\Users\\Liuxk\\.gemini\\antigravity\\brain\\e8d9ff57-4a93-4281-aae8-6a3054d5887f\\preview_coverage.png"
    plt.savefig(out_path, dpi=200)
    print(f"Preview saved to {out_path}")

if __name__ == "__main__":
    preview_serial()
