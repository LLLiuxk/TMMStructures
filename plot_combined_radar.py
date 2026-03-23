import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def compute_polar_properties(C_eff, num_points=360):
    """
    Computes directional Young's Modulus E(theta) from effective stiffness matrix.
    Adds a tiny regularization to ensure stability for non-connected structures.
    """
    # 1. Add tiny regularization to prevent singular matrix inversion for non-connected designs
    C_stable = C_eff + np.eye(3) * 1e-9
    
    try:
        S = np.linalg.inv(C_stable)
    except np.linalg.LinAlgError:
        return None, None
        
    S11, S22 = S[0, 0], S[1, 1]
    S12, S66 = S[0, 1], S[2, 2]
    S16, S26 = S[0, 2], S[1, 2]
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    E_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)
        # Standard 2D compliance rotation for E(theta)
        # inv_E = S11*c^4 + S22*s^4 + (2*S12 + S66)*c^2*s^2 + 2*S16*c^3*s + 2*S26*c*s^3
        inv_E = S11 * c**4 + S22 * s**4 + (2*S12 + S66) * c**2 * s**2 + 2*S16 * c**3 * s + 2*S26 * c * s**3
        
        # Clamp inv_E to avoid division by zero or negative values due to numerical noise
        E_theta[i] = 1.0 / max(inv_E, 1e-12)
            
    # Close the loop
    thetas = np.append(thetas, thetas[0])
    E_theta = np.append(E_theta, E_theta[0])
    return thetas, E_theta

def compute_thermal_polar_properties(kappa_eff, num_points=360):
    """
    Computes directional Thermal Conductivity kappa(theta).
    """
    k11, k22, k12 = kappa_eff[0, 0], kappa_eff[1, 1], kappa_eff[0, 1]
    
    thetas = np.linspace(0, 2*np.pi, num_points)
    kappa_theta = np.zeros_like(thetas)
    
    for i, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)
        # Kappa(theta) = k_ij * n_i * n_j
        kappa_val = k11 * c**2 + k22 * s**2 + 2 * k12 * c * s
        kappa_theta[i] = max(kappa_val, 1e-12)
            
    thetas = np.append(thetas, thetas[0])
    kappa_theta = np.append(kappa_theta, kappa_theta[0])
    return thetas, kappa_theta

def save_radar_chart(C_eff, kappa_eff, title, out_file):
    """
    Renders a dual-subplot radar chart for Mechanical and Thermal properties.
    """
    thetas_E, E_theta = compute_polar_properties(C_eff)
    thetas_k, kappa_theta = compute_thermal_polar_properties(kappa_eff)
    
    if thetas_E is None: 
        print(f"Warning: Skipping radar plot for {title} due to singular matrix.")
        return
        
    # Use 1x2 subplot layout for separation
    fig = plt.figure(figsize=(12, 6))
    
    # --- Subplot 1: Mechanical (Young's Modulus) ---
    ax1 = fig.add_subplot(121, polar=True)
    max_E = np.nanmax(E_theta)
    
    if max_E > 1e-10:
        # Use log scale normalization to handle extreme anisotropy
        E_log = np.log10(np.maximum(E_theta, 1e-12))
        max_log = np.max(E_log)
        min_log = np.min(E_log)
        # Scale to [0.1, 1.0] so the minimum is still slightly visible (not just a single dot at center)
        if max_log > min_log:
            E_norm = (E_log - min_log) / (max_log - min_log) * 0.9 + 0.1
        else:
            E_norm = np.ones_like(E_theta)
    else:
        E_norm = E_theta
    
    ax1.plot(thetas_E, E_norm, linewidth=2, color='royalblue')
    ax1.fill(thetas_E, E_norm, alpha=0.3, color='royalblue')
    ax1.set_title(f"Young's Modulus $E(\\theta)$\nMax: {max_E:.3e}", fontsize=11, pad=15)
    ax1.set_yticks([0.1, 0.55, 1.0])
    ax1.set_yticklabels(['min', 'log-mid', 'max'], fontsize=8)

    # --- Subplot 2: Thermal (Conductivity) ---
    ax2 = fig.add_subplot(122, polar=True)
    max_kappa = np.nanmax(kappa_theta)
    
    if max_kappa > 1e-10:
        k_log = np.log10(np.maximum(kappa_theta, 1e-12))
        max_log_k = np.max(k_log)
        min_log_k = np.min(k_log)
        if max_log_k > min_log_k:
            kappa_norm = (k_log - min_log_k) / (max_log_k - min_log_k) * 0.9 + 0.1
        else:
            kappa_norm = np.ones_like(kappa_theta)
    else:
        kappa_norm = kappa_theta
    
    ax2.plot(thetas_k, kappa_norm, linewidth=2, color='crimson')
    ax2.fill(thetas_k, kappa_norm, alpha=0.3, color='crimson')
    ax2.set_title(f"Thermal Cond. $\\kappa(\\theta)$\nMax: {max_kappa:.3e}", fontsize=11, pad=15)
    ax2.set_yticks([0.1, 0.55, 1.0])
    ax2.set_yticklabels(['min', 'log-mid', 'max'], fontsize=8)

    fig.suptitle(title, fontsize=14, weight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test script for manual verification
    import glob
    # (Assuming homogenize structure logic remains the same)
    print("This script is now a utility for generate_dataset.py. Use reproduce.bat to test individual samples.")
