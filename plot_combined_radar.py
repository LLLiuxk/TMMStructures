import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'D:\PyProjects\TMMStructures')
from homogenize import load_and_reconstruct, homogenize_elastic, homogenize_thermal

def compute_polar_properties(C_eff, num_points=360):
    try:
        S = np.linalg.inv(C_eff)
    except np.linalg.LinAlgError:
        return None, None
        
    S11, S22 = S[0, 0], S[1, 1]
    S12, S66 = S[0, 1], S[2, 2]
    S16, S26 = S[0, 2], S[1, 2]
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    E_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)
        inv_E = S11 * c**4 + S22 * s**4 + (2*S12 + S66) * c**2 * s**2 + 2*S16 * c**3 * s + 2*S26 * c * s**3
        E_theta[i] = 1.0 / inv_E if inv_E > 0 else np.nan
            
    thetas = np.append(thetas, thetas[0])
    E_theta = np.append(E_theta, E_theta[0])
    return thetas, E_theta

def compute_thermal_polar_properties(kappa_eff, num_points=360):
    k11, k22, k12 = kappa_eff[0, 0], kappa_eff[1, 1], kappa_eff[0, 1]
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    kappa_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)
        kappa_theta[i] = k11 * c**2 + k22 * s**2 + 2 * k12 * c * s
            
    thetas = np.append(thetas, thetas[0])
    kappa_theta = np.append(kappa_theta, kappa_theta[0])
    return thetas, kappa_theta

def main():
    folder = r"D:\PyProjects\TMMStructures\Output\sample_structures"
    if not os.path.exists(folder):
        return
        
    for filename in os.listdir(folder):
        if filename.endswith(".png") and not "radar" in filename:
            filepath = os.path.join(folder, filename)
            print(f"Processing Combined: {filename}")
            
            density = load_and_reconstruct(filepath, invert=True)
            nely, nelx = density.shape
            
            C_eff = homogenize_elastic(nelx, nely, density, E0=1.0, Emin=1e-9, nu=0.3, penal=1.0)
            kappa_eff = homogenize_thermal(nelx, nely, density, k0=1.0, kmin=1e-9, penal=1.0)
            
            thetas_E, E_theta = compute_polar_properties(C_eff)
            thetas_k, kappa_theta = compute_thermal_polar_properties(kappa_eff)
            
            if thetas_E is None:
                continue
                
            fig = plt.figure(figsize=(7, 7))
            
            # Setup dual axes in polar coordinates
            ax_E = fig.add_subplot(111, polar=True)
            
            # Plot Mechanical Property (E) on ax_E
            line_E, = ax_E.plot(thetas_E, E_theta, linewidth=2.5, color='royalblue', label=r"Young's Modulus $E(\theta)$")
            ax_E.fill(thetas_E, E_theta, alpha=0.2, color='royalblue')
            ax_E.set_ylabel("Young's Modulus E", color='royalblue', labelpad=40, weight='bold')
            ax_E.tick_params(axis='y', colors='royalblue')
            
            # Create a second twin polar axis for Thermal Conductivity (kappa)
            # Since matplotlib doesn't natively support polar twin axes perfectly with different r-scales sometimes,
            # Normalizing/Scaling might be needed, or using secondary_y trick on standard plot.
            # However, for polar plots, standard approach is simply to plot on the same axis if ranges are similar,
            # or normalized plotting. Let's normalize both to [0, 1] relative to their maximums for best visual comparison
            # of shape and anisotropy on the same chart, and write the max values in the legend or title.
            
            max_E = np.nanmax(E_theta)
            max_kappa = np.nanmax(kappa_theta)
            
            E_norm = E_theta / max_E
            kappa_norm = kappa_theta / max_kappa
            
            ax_E.clear() # Clear and redraw normalized
            
            line_E, = ax_E.plot(thetas_E, E_norm, linewidth=2.5, color='royalblue', label=f"Young's Modulus $E(\theta)$\n(Max: {max_E:.3f})")
            ax_E.fill(thetas_E, E_norm, alpha=0.2, color='royalblue')
            
            line_k, = ax_E.plot(thetas_k, kappa_norm, linewidth=2.5, color='crimson', label=f"Thermal Cond. $\kappa(\theta)$\n(Max: {max_kappa:.3f})")
            ax_E.fill(thetas_k, kappa_norm, alpha=0.15, color='crimson')
            
            ax_E.set_title(f"Normalized Combined Radar Chart\n{filename}", pad=20, fontsize=14, weight='bold')
            ax_E.set_ylim(bottom=0, top=1.1)
            ax_E.grid(True, linestyle='--', alpha=0.6)
            
            # Set grid labels
            ax_E.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_E.set_yticklabels(['25%', '50%', '75%', '100%'], color='dimgray', fontsize=9)
            ax_E.set_rlabel_position(22.5) 
            plt.xticks(fontsize=10)
            
            # Legend
            ax_E.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10, frameon=True)
            
            out_file = os.path.join(folder, filename.replace(".png", "_combined_radar.png"))
            plt.tight_layout()
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined radar chart to {out_file}\n")

if __name__ == "__main__":
    main()
