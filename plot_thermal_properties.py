import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'D:\PyProjects\TMMStructures')
from homogenize import load_and_reconstruct, homogenize_thermal

def compute_thermal_polar_properties(kappa_eff, num_points=360):
    """
    Computes thermal conductivity kappa(theta) from effective conductivity tensor kappa_eff.
    kappa(theta) = n^T * kappa_eff * n, where n = [cos(theta), sin(theta)]^T
    """
    k11 = kappa_eff[0, 0]
    k22 = kappa_eff[1, 1]
    k12 = kappa_eff[0, 1] # assuming symmetric k12 = k21
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    kappa_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c = np.cos(theta)
        s = np.sin(theta)
        
        # kappa_theta[i] = k11*c^2 + k22*s^2 + 2*k12*s*c
        kappa_theta[i] = k11 * c**2 + k22 * s**2 + 2 * k12 * c * s
            
    # duplicate the first point at the end to close the loop
    thetas = np.append(thetas, thetas[0])
    kappa_theta = np.append(kappa_theta, kappa_theta[0])
            
    return thetas, kappa_theta

def main():
    folder = r"D:\PyProjects\TMMStructures\Output\sample_structures"
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return
        
    for filename in os.listdir(folder):
        # process only the original generated structures
        if filename.endswith(".png") and not filename.endswith("_radar.png") and not filename.endswith("_thermal_radar.png"):
            filepath = os.path.join(folder, filename)
            print(f"Processing Thermal: {filename}")
            
            density = load_and_reconstruct(filepath, invert=True)
            nely, nelx = density.shape
            
            # Penal=1 for linear mapping of binary image
            kappa_eff = homogenize_thermal(nelx, nely, density, k0=1.0, kmin=1e-9, penal=1.0)
            print("kappa_eff:")
            print(kappa_eff)
            
            thetas, kappa_theta = compute_thermal_polar_properties(kappa_eff)
            
            # Plot Polar
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
            
            ax.plot(thetas, kappa_theta, linewidth=2, color='crimson', label=r"$\kappa(\theta)$")
            ax.fill(thetas, kappa_theta, alpha=0.3, color='lightcoral')
            
            ax.set_title(f"Radar Chart: Thermal Conductivity\n{filename}", pad=20, fontsize=14)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=9)
            
            out_file = os.path.join(folder, filename.replace(".png", "_thermal_radar.png"))
            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved thermal radar chart to {out_file}\n")

if __name__ == "__main__":
    main()
