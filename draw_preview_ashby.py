import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_hs_bounds(vf_range, E0=1.0, nu0=0.3):
    K0 = E0 / (2 * (1 - nu0))
    G0 = E0 / (2 * (1 + nu0))
    K_up, G_up = [], []
    for f in vf_range:
        k_hs = K0 + (1-f) / (1/(0 - K0) + f/(K0 + G0))
        g_hs = G0 + (1-f) / (1/(0 - G0) + f/(G0 * (K0 + 2*G0) / (2 * K0 + 2*G0)))
        K_up.append(k_hs)
        G_up.append(g_hs)
    return np.array(K_up), np.array(G_up)

# Load the saved 50-sample preview
with open('preview_data.json', 'r') as f:
    data = json.load(f)

vfs = np.array(data['vfs'])
c11s = np.array(data['c11s'])
ks = np.array(data['ks'])

vf_plot = np.linspace(0.001, 1, 100)
K_up, G_up = calculate_hs_bounds(vf_plot)
E_up = 9 * K_up * G_up / (3 * K_up + G_up)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: C11 Ashby Plot (Log-Log)
ax1.plot(vf_plot, E_up, 'r-', linewidth=2, label='HS Upper Bound (E)')
ax1.scatter(vfs, c11s, c='blue', alpha=0.7, s=40, label='Generated Microstructures (Preview)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim([0.05, 1.0])
ax1.set_ylim([1e-4, 2.0])
ax1.set_xlabel('Volume Fraction $\phi$', fontsize=12)
ax1.set_ylabel('Stiffness $C_{11}$', fontsize=12)
ax1.set_title('Log-Log Ashby Plot: Stiffness vs $\phi$', fontsize=14, weight='bold')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Bulk Modulus (K) Ashby Plot (Log-Log)
ax2.plot(vf_plot, K_up, 'g-', linewidth=2, label='HS Upper Bound (K)')
ax2.scatter(vfs, ks, c='green', alpha=0.7, s=40, label='Generated Microstructures (Preview)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim([0.05, 1.0])
ax2.set_ylim([1e-4, 2.0])
ax2.set_xlabel('Volume Fraction $\phi$', fontsize=12)
ax2.set_ylabel('Bulk Modulus $K$', fontsize=12)
ax2.set_title('Log-Log Ashby Plot: Bulk Modulus vs $\phi$', fontsize=14, weight='bold')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
out_png = "C:\\Users\\Liuxk\\.gemini\\antigravity\\brain\\e8d9ff57-4a93-4281-aae8-6a3054d5887f\\property_coverage_summary.png"
plt.savefig(out_png, dpi=300)
print(f"Ashby plots saved to {out_png}")
