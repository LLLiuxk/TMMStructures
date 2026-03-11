import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'D:\PyProjects\TMMStructures')
from homogenize import load_and_reconstruct, homogenize_elastic

def compute_polar_properties(C_eff, num_points=360):
    """
    Computes Young's modulus E(theta) from effective stiffness matrix C_eff.
    """
    try:
        S = np.linalg.inv(C_eff)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered.")
        return None, None
        
    S11 = S[0, 0]
    S22 = S[1, 1]
    S12 = S[0, 1]
    S66 = S[2, 2]
    S16 = S[0, 2]
    S26 = S[1, 2]
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    E_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c = np.cos(theta) # x-axis mapping
        s = np.sin(theta) # y-axis mapping
        
        inv_E = S11 * c**4 + S22 * s**4 + (2*S12 + S66) * c**2 * s**2 + 2*S16 * c**3 * s + 2*S26 * c * s**3
        if inv_E > 0:
            E_theta[i] = 1.0 / inv_E
        else:
            E_theta[i] = np.nan
            
    # duplicate the first point at the end to close the loop
    thetas = np.append(thetas, thetas[0])
    E_theta = np.append(E_theta, E_theta[0])
            
    return thetas, E_theta

def main():
    folder = r"D:\PyProjects\TMMStructures\Output\sample_structures"
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return
        
    for filename in os.listdir(folder):
        if filename.endswith(".png") and not filename.endswith("_radar.png"):
            filepath = os.path.join(folder, filename)
            print(f"Processing: {filename}")
            
            density = load_and_reconstruct(filepath, invert=True)
            nely, nelx = density.shape
            
            # Penal=1 ensures binary 0/1 inputs are mapped linearly to material properties.
            C_eff = homogenize_elastic(nelx, nely, density, E0=1.0, Emin=1e-9, nu=0.3, penal=1.0)
            print("C_eff:")
            print(C_eff)
            
            thetas, E_theta = compute_polar_properties(C_eff)
            if thetas is None:
                continue
                
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
            
            ax.plot(thetas, E_theta, linewidth=2, color='royalblue', label=r"$E(\theta)$")
            ax.fill(thetas, E_theta, alpha=0.3, color='cornflowerblue')
            
            ax.set_title(f"Radar Chart: Young's Modulus\n{filename}", pad=20, fontsize=14)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=9)
            
            out_file = os.path.join(folder, filename.replace(".png", "_radar.png"))
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved radar chart to {out_file}\n")

if __name__ == "__main__":
    main()
