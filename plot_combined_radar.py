import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def save_radar_chart(C_eff, kappa_eff, title, out_file):
    thetas_E, E_theta = compute_polar_properties(C_eff)
    thetas_k, kappa_theta = compute_thermal_polar_properties(kappa_eff)
    
    if thetas_E is None: return
        
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    
    max_E = np.nanmax(E_theta)
    max_kappa = np.nanmax(kappa_theta)
    
    E_norm = E_theta / max_E if max_E > 0 else E_theta
    kappa_norm = kappa_theta / max_kappa if max_kappa > 0 else kappa_theta
    
    ax.plot(thetas_E, E_norm, linewidth=2.5, color='royalblue', label=f"Young's Modulus $E(\\theta)$\n(Max: {max_E:.3f})")
    ax.fill(thetas_E, E_norm, alpha=0.2, color='royalblue')
    
    ax.plot(thetas_k, kappa_norm, linewidth=2.5, color='crimson', label=f"Thermal Cond. $\\kappa(\\theta)$\n(Max: {max_kappa:.3f})")
    ax.fill(thetas_k, kappa_norm, alpha=0.15, color='crimson')
    
    ax.set_title(title, pad=20, fontsize=12, weight='bold')
    ax.set_ylim(bottom=0, top=1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='dimgray', fontsize=9)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Backwards compatibility for the original script logic
    import glob
    from homogenize import load_and_reconstruct, homogenize_elastic, homogenize_thermal
    
    folder = r"D:\PyProjects\TMMStructures\Output\sample_structures"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".png") and not "radar" in filename:
                filepath = os.path.join(folder, filename)
                density = load_and_reconstruct(filepath, invert=True)
                nely, nelx = density.shape
                C_eff = homogenize_elastic(nelx, nely, density, penal=1.0)
                kappa_eff = homogenize_thermal(nelx, nely, density, penal=1.0)
                out_file = os.path.join(folder, filename.replace(".png", "_combined_radar.png"))
                save_radar_chart(C_eff, kappa_eff, f"Combined Radar: {filename}", out_file)
