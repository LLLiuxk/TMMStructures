"""
Diagnostic: sample_0000 half-line radar chart issue analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================
# 1. sample_0000 actual data (from dataset_schema.json)
# =====================================================
C_eff_raw = np.array([
    [0.0365490823612805,    2.0232361759394445e-09,  6.511473142776192e-10],
    [2.0232361759394445e-09, 5.235038727229114e-09, -3.6641593850213263e-10],
    [6.511473142776192e-10, -3.6641593850213263e-10, 2.87373102064661e-09]
])

kappa_eff_raw = np.array([
    [0.1368185714508198,    1.3154834352029457e-09],
    [1.3154834352029457e-09, 5.923493640693008e-09]
])

print("=" * 60)
print("  sample_0000 Diagnosis Report")
print("=" * 60)
print()
print("[Mechanical Matrix C_eff]:")
print("  C11 = {:.6e}  <- Normal, X-direction has solid path".format(C_eff_raw[0,0]))
print("  C22 = {:.6e}  <- Nearly ZERO! No Y-direction solid path".format(C_eff_raw[1,1]))
print("  C12 = {:.6e}".format(C_eff_raw[0,1]))
print("  C66 = {:.6e}".format(C_eff_raw[2,2]))
print()
print("  WARNING: Anisotropy ratio C11/C22 = {:.2e}  (7 orders of magnitude!)".format(
    C_eff_raw[0,0] / C_eff_raw[1,1]))
print()
print("[Thermal Matrix kappa_eff]:")
print("  k11 = {:.6e}  <- Normal".format(kappa_eff_raw[0,0]))
print("  k22 = {:.6e}  <- Nearly ZERO! No Y-direction heat path".format(kappa_eff_raw[1,1]))
print("  WARNING: k11/k22 = {:.2e}  (7 orders of magnitude!)".format(
    kappa_eff_raw[0,0] / kappa_eff_raw[1,1]))
print()

# =====================================================
# 2. Compute polar curves (same logic as plot_combined_radar.py)
# =====================================================
def compute_E_theta(C_eff, num_pts=360):
    C_stable = C_eff + np.eye(3) * 1e-9
    S = np.linalg.inv(C_stable)
    S11, S22 = S[0,0], S[1,1]
    S12, S66 = S[0,1], S[2,2]
    S16, S26 = S[0,2], S[1,2]
    thetas = np.linspace(0, 2*np.pi, num_pts)
    E_vals = []
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        inv_E = S11*c**4 + S22*s**4 + (2*S12+S66)*c**2*s**2 + 2*S16*c**3*s + 2*S26*c*s**3
        E_vals.append(1.0 / max(inv_E, 1e-12))
    thetas = np.append(thetas, thetas[0])
    E_vals = np.append(E_vals, E_vals[0])
    return thetas, np.array(E_vals)

def compute_kappa_theta(kappa_eff, num_pts=360):
    k11, k22, k12 = kappa_eff[0,0], kappa_eff[1,1], kappa_eff[0,1]
    thetas = np.linspace(0, 2*np.pi, num_pts)
    k_vals = []
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        k_vals.append(max(k11*c**2 + k22*s**2 + 2*k12*c*s, 1e-12))
    thetas = np.append(thetas, thetas[0])
    k_vals = np.append(k_vals, k_vals[0])
    return thetas, np.array(k_vals)

thetas_E, E_vals = compute_E_theta(C_eff_raw)
thetas_k, k_vals = compute_kappa_theta(kappa_eff_raw)

print("[E(theta) Sampling Analysis]:")
for deg in [0, 30, 60, 90, 120, 150, 180]:
    idx = int(deg / 360 * 360)
    print("  E({:3d}deg) = {:.6e}  (normalized={:.4f})".format(
        deg, E_vals[idx], E_vals[idx]/E_vals.max()))
print()
print("Result: E(90deg)/E(0deg) = {:.2e}".format(E_vals[90]/E_vals[0]))
print("-> After linear normalization, 90-deg direction ~ 0 -> half-line visual artifact")

# =====================================================
# 3. Diagnostic figure: current vs. improved
# =====================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle("sample_0000 Diagnosis: Why is the Mechanical Radar Plot a Half-Line?",
             fontsize=14, weight='bold', y=0.98)

# --- Subplot 1: Current visualization (normalized, the problem) ---
ax1 = fig.add_subplot(231, polar=True)
E_norm = E_vals / E_vals.max()
ax1.plot(thetas_E, E_norm, 'royalblue', lw=2)
ax1.fill(thetas_E, E_norm, alpha=0.3, color='royalblue')
ax1.set_title("Current: Young's Modulus\n(linear normalized - PROBLEM)", fontsize=10, pad=15)
ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
ax1.set_yticklabels(['25%','50%','75%','100%'], fontsize=7)

# --- Subplot 2: Log-scale visualization (reveals true anisotropy) ---
ax2 = fig.add_subplot(232, polar=True)
E_log = np.log10(np.maximum(E_vals, 1e-15))
E_log_shifted = E_log - E_log.min()
ax2.plot(thetas_E, E_log_shifted, 'darkorange', lw=2)
ax2.fill(thetas_E, E_log_shifted, alpha=0.3, color='darkorange')
ax2.set_title("Improved: log10[E(theta)]\n(reveals all directions visibly)", fontsize=10, pad=15)
ax2.set_yticks([])

# --- Subplot 3: Structure connectivity diagram ---
ax3 = fig.add_subplot(233)
ax3.set_xlim(-0.5, 1.5)
ax3.set_ylim(-0.5, 1.5)
ax3.set_aspect('equal')
ax3.set_title("sample_0000 Connectivity\n(why C22 is nearly zero)", fontsize=10)

rect = plt.Rectangle((0,0), 1, 1, fill=False, ec='black', lw=2)
ax3.add_patch(rect)

# Nodes: E1=Left, E2=Bottom, E3=Right, E4=Top
# Actual node positions from schema:
# E1: pos=0.12 on left edge (y=1-0.12=0.88 if top=1)
# E2: pos=0.2116 on bottom edge
# E3: pos=0.5724 on right edge
# E4: pos=0.2749 on top edge
nodes = {
    'E1 (left)': (0.0, 1-0.12),
    'E2 (bottom)': (0.2116, 0.0),
    'E3 (right)': (1.0, 1-0.5724),
    'E4 (top)': (0.2749, 1.0),
}
node_keys = list(nodes.keys())
colors = ['blue', 'green', 'red', 'purple']

for (name, (x, y)), col in zip(nodes.items(), colors):
    ax3.plot(x, y, 'o', markersize=10, color=col, zorder=5)
    offset_x = 0.08 if x < 0.5 else -0.25
    offset_y = 0.07
    ax3.annotate(name, (x, y), xytext=(x+offset_x, y+offset_y), fontsize=8, color=col)

# Connections: E1-E2 (straight), E1-E3 (straight), E2-E3 (bezier), E3-E4 (circular_arc)
node_coords = list(nodes.values())
conn_pairs = [(0,1,'straight'), (0,2,'straight'), (1,2,'bezier'), (2,3,'arc')]
conn_colors_list = ['royalblue', 'royalblue', 'darkorange', 'crimson']
for (ia, ib, ctype), cc in zip(conn_pairs, conn_colors_list):
    xa, ya = node_coords[ia]; xb, yb = node_coords[ib]
    ax3.plot([xa, xb], [ya, yb], '-', color=cc, lw=2.5, alpha=0.8, label=ctype)

ax3.annotate('', xy=(1.4, 0.5), xytext=(1.1, 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(1.42, 0.5, 'X\n(C11 large)', color='red', fontsize=8, va='center')
ax3.annotate('', xy=(0.5, 1.4), xytext=(0.5, 1.1),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.text(0.5, 1.44, 'Y (C22~0)', color='blue', fontsize=8, ha='center')
ax3.set_xticks([]); ax3.set_yticks([])
ax3.text(0.5, -0.3, 'X-path: E1->E3 strong\nY-path: no direct vertical link',
         ha='center', fontsize=9, color='darkgreen')

# --- Subplot 4: Log-normalized mechanical radar ---
ax4 = fig.add_subplot(234, polar=True)
E_log2 = np.log10(np.maximum(E_vals, 1e-12))
rng = E_log2.max() - E_log2.min()
E_log_norm = (E_log2 - E_log2.min()) / (rng + 1e-15)
ax4.plot(thetas_E, E_log_norm, 'royalblue', lw=2)
ax4.fill(thetas_E, E_log_norm, alpha=0.3, color='royalblue')
ax4.set_title("Fixed: log-normalized E(theta)\nMax E: {:.3e}".format(E_vals.max()), fontsize=10, pad=15)
ax4.set_yticks([0.25, 0.5, 0.75, 1.0])

# --- Subplot 5: Thermal kappa (same problem) ---
ax5 = fig.add_subplot(235, polar=True)
k_norm = k_vals / k_vals.max()
ax5.plot(thetas_k, k_norm, 'crimson', lw=2)
ax5.fill(thetas_k, k_norm, alpha=0.3, color='crimson')
ax5.set_title("Thermal kappa(theta) - same issue\nk11/k22={:.0e}".format(
    kappa_eff_raw[0,0]/kappa_eff_raw[1,1]), fontsize=10, pad=15)

# --- Subplot 6: Text summary ---
ax6 = fig.add_subplot(236)
ax6.axis('off')
summary_lines = [
    "DIAGNOSIS SUMMARY",
    "",
    "Is the data correct?",
    "  YES. Values accurately reflect",
    "  the physical reality.",
    "",
    "Root Cause (Two Layers):",
    "",
    "1. PHYSICS (real phenomenon):",
    "   The structure only forms a",
    "   connected load path in the",
    "   X direction (E1->E3).",
    "   C11 = 0.0365  (real stiffness)",
    "   C22 = 5e-9    (= Emin, real!)",
    "",
    "2. VISUALIZATION (display issue):",
    "   Linear normalization compresses",
    "   weak directions to zero,",
    "   causing the half-line effect.",
    "",
    "FIX: Use log-normalization (left)",
    "to make weak directions visible.",
]
ax6.text(0.05, 0.97, "\n".join(summary_lines), transform=ax6.transAxes,
         fontsize=9, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
out_path = "Output/sample_0000_diagnosis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print()
print("Diagnosis figure saved to: {}".format(out_path))
