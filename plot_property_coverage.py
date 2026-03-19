import json
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_hs_bounds(vf_range, E0=1.0, nu0=0.3):
    """
    Calculates Hashin-Shtrikman bounds for 2D plane stress.
    For a composite of material (E0, nu0) and void.
    Bulk modulus K = E / (2*(1-nu))
    Shear modulus G = E / (2*(1+nu))
    """
    K0 = E0 / (2 * (1 - nu0))
    G0 = E0 / (2 * (1 + nu0))
    
    # Lower bounds for void composite are zero
    # Upper bounds:
    K_hs = []
    G_hs = []
    
    for f in vf_range:
        if f <= 0:
            K_hs.append(0); G_hs.append(0)
            continue
        # HS Upper Bound formulas
        k_up = K0 + (1-f) / (1/(0 - K0) + f/(K0 + G0))
        g_up = G0 + (1-f) / (1/(0 - G0) + f/(G0 * (K0 + 2*G0) / (2 * K0 + 2*G0)))
        
        K_hs.append(k_up)
        G_hs.append(g_up)
        
    return np.array(K_hs), np.array(G_hs)

def main():
    config_path = "dataset_config.json"
    out_dir = "Output/dataset"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            out_dir = config.get("output_dir", out_dir)
            
    json_path = os.path.join(out_dir, "dataset_schema.json")
    if not os.path.exists(json_path):
        print("Dataset schema not found.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    vfs = []
    c11s = []
    c22s = []
    k11s = []
    
    valid_count = 0
    for record in data:
        if 'properties' not in record:
            continue
            
        p = record['properties']
        vfs.append(p['volume_fraction'])
        c11s.append(p['C11'])
        c22s.append(p['C22'])
        k11s.append(p['k11'])
        valid_count += 1
        
    if valid_count == 0:
        print("Error: No valid records with property data found in dataset_schema.json.")
        print("Please delete 'Output/dataset/' and run generate.bat again to rebuild the dataset.")
        return

    print(f"Analyzing {valid_count} valid records...")
    vfs = np.array(vfs)
    c11s = np.array(c11s)
    c22s = np.array(c22s)
    
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: C11 vs Volume Fraction with HS Bound ---
    vf_plot = np.linspace(0, 1, 100)
    K_hs, G_hs = calculate_hs_bounds(vf_plot)
    # In 2D plane stress, C11 upper bound is related to HS
    # For isotropic, C11 = K + G. 
    c11_hs_up = K_hs + G_hs
    
    ax1.plot(vf_plot, c11_hs_up, 'r--', label='HS Upper Bound (Isotropic Reference)')
    ax1.scatter(vfs, c11s, alpha=0.6, edgecolors='w', label='Generated Microstructures')
    
    ax1.set_xlabel('Volume Fraction $\phi$', fontsize=12)
    ax1.set_ylabel('Effective Stiffness $C_{11}$', fontsize=12)
    ax1.set_title('Mechanical Property Coverage', fontsize=14, weight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()
    
    # --- Plot 2: C11 vs C22 Anisotropy Plot ---
    max_val = max(max(c11s), max(c22s)) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Isotropic Line')
    ax2.scatter(c11s, c22s, c=vfs, cmap='viridis', alpha=0.8, edgecolors='none')
    
    ax2.set_xlabel('$C_{11}$', fontsize=12)
    ax2.set_ylabel('$C_{22}$', fontsize=12)
    ax2.set_title('Anisotropy & Range Coverage', fontsize=14, weight='bold')
    colorbar = plt.colorbar(ax2.collections[0], ax=ax2)
    colorbar.set_label('Volume Fraction', rotation=270, labelpad=15)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "property_coverage_summary.png")
    plt.savefig(out_path, dpi=300)
    print(f"Property coverage plot saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
