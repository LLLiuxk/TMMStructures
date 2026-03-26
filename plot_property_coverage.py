import json
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_hs_bounds(vf_range, E0=1.0, nu0=0.3):
    """
    Calculates Hashin-Shtrikman upper bounds for 2D plane stress.
    For a composite of material (E0, nu0) and void.
    """
    K0 = E0 / (2 * (1 - nu0))
    G0 = E0 / (2 * (1 + nu0))
    K_hs = []
    G_hs = []
    for f in vf_range:
        if f <= 0:
            K_hs.append(0); G_hs.append(0)
            continue
        # Phase 1 is void (K=0, G=0)
        k_up = K0 + (1-f) / (1/(0 - K0) + f/(K0 + G0))
        g_up = G0 + (1-f) / (1/(0 - G0) + f/(G0 * (K0 + 2*G0) / (2 * K0 + 2*G0)))
        K_hs.append(k_up)
        G_hs.append(g_up)
    return np.array(K_hs), np.array(G_hs)

def generate_coverage_plot(out_dir, show_plot=False):
    json_path = os.path.join(out_dir, "dataset_schema.json")
    if not os.path.exists(json_path):
        print(f"Error: Dataset schema not found at {json_path}")
        return
        
    print(f"Loading {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    vfs, c11s, c22s, c12s, ks = [], [], [], [], []
    
    for record in data:
        if 'properties' not in record or 'C11' not in record['properties']: continue
        p = record['properties']
        vfs.append(p['volume_fraction'])
        c11s.append(p['C11'])
        c22s.append(p['C22'])
        c12s.append(p.get('C12', 0))
        # K = (C11 + C22 + 2*C12) / 4 for 2D
        ks.append((p['C11'] + p['C22'] + 2*p.get('C12', 0)) / 4.0)
        
    valid_count = len(vfs)
    if valid_count == 0:
        print("Error: No valid property data found.")
        return

    vfs, c11s, c22s, ks = np.array(vfs), np.array(c11s), np.array(c22s), np.array(ks)
    vf_plot = np.linspace(0.001, 1, 100)
    K_hs_up, G_hs_up = calculate_hs_bounds(vf_plot)
    
    # 2x2 Plot Layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. C11 vs VF (Linear Adaptive)
    ax = axs[0,0]
    ax.plot(vf_plot, K_hs_up + G_hs_up, 'r--', label='HS Upper Bound ($C_{11}$)')
    ax.scatter(vfs, c11s, s=8, alpha=0.15, edgecolors='none', label='Data')
    ax.set_ylim(0, max(c11s)*1.2 if len(c11s)>0 else 1)
    ax.set_title("A. $C_{11}$ vs Volume Fraction (Linear)", fontsize=13, weight='bold')
    ax.set_xlabel('$\phi$'); ax.set_ylabel('$C_{11}$'); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Bulk Modulus K vs VF (Log-Log / Ashby)
    ax = axs[0,1]
    ax.plot(vf_plot, K_hs_up, 'r--', label='HS Upper Bound ($K$)')
    ax.scatter(vfs, ks, s=8, alpha=0.15, edgecolors='none', color='forestgreen')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.05, 1.0); ax.set_ylim(1e-5, 2)
    ax.set_title("B. Bulk Modulus $K$ vs $\phi$ (Log-Log Ashby)", fontsize=13, weight='bold')
    ax.set_xlabel('$\log \phi$'); ax.set_ylabel('$\log K$'); ax.grid(True, which="both", alpha=0.3)

    # 3. Anisotropy C11 vs C22 (Linear)
    ax = axs[1,0]
    m_val = max(max(c11s), max(c22s)) * 1.1
    ax.plot([0, m_val], [0, m_val], 'k--', alpha=0.5, label='Isotropic')
    sc = ax.scatter(c11s, c22s, c=vfs, s=8, cmap='viridis', alpha=0.2, edgecolors='none')
    ax.set_title("C. Anisotropy $C_{11}$ vs $C_{22}$ (Linear)", fontsize=13, weight='bold')
    ax.set_xlabel('$C_{11}$'); ax.set_ylabel('$C_{22}$'); plt.colorbar(sc, ax=ax, label='$\phi$')

    # 4. Anisotropy C11 vs C22 (Log-Log)
    ax = axs[1,1]
    ax.plot([1e-5, 2], [1e-5, 2], 'k--', alpha=0.5, label='Isotropic')
    sc2 = ax.scatter(c11s, c22s, c=vfs, s=8, cmap='plasma', alpha=0.2, edgecolors='none')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-5, 2); ax.set_ylim(1e-5, 2)
    ax.set_title("D. Anisotropy (Log-Log)", fontsize=13, weight='bold')
    ax.set_xlabel('$\log C_{11}$'); ax.set_ylabel('$\log C_{22}$')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "property_coverage_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Academic property summary saved to {out_path}")
    if show_plot: plt.show()
    plt.close()

def main():
    config_path = "dataset_config.json"
    out_dir = "Output/dataset/batch_1" # Default to batch_1 for safety
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            out_dir = config.get("output_dir", out_dir)
    generate_coverage_plot(out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", nargs="?", default="Output/dataset/batch_1", help="Target directory")
    args = parser.parse_args()
    
    generate_coverage_plot(args.target_dir)
