"""
Deep homogenization test for sample_0000
==========================================
Tests the actual homogenization of the 128x128 image and a mirrored version
to see which gives physically reasonable results.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(0, 'D:\\PyProjects\\TMMStructures')
from homogenize import homogenize_elastic, homogenize_thermal

print("=" * 65)
print("  DEEP HOMOGENIZATION TEST: sample_0000")
print("=" * 65)

# Load raw image
img_path = "Output/dataset/batch_1/images/sample_0000.png"
img = Image.open(img_path).convert('L')
density_raw = np.array(img, dtype=np.float64) / 255.0
# Black = solid (value near 0), White = void (value near 255/255=1)
density_quarter = (density_raw < 0.5).astype(np.float64)
h, w = density_quarter.shape
print(f"\nQuarter image: {w}x{h}, solid fraction = {density_quarter.mean():.4f}")

# Check connectivity
from scipy.ndimage import label
labeled, ncomp = label(density_quarter)
print(f"Number of connected solid components: {ncomp}")
top_labels = set(labeled[0, :][labeled[0, :] > 0])
bot_labels = set(labeled[h-1, :][labeled[h-1, :] > 0])
left_labels = set(labeled[:, 0][labeled[:, 0] > 0])
right_labels = set(labeled[:, w-1][labeled[:, w-1] > 0])
print(f"  Top edge solid labels:    {top_labels}")
print(f"  Bottom edge solid labels: {bot_labels}")
print(f"  Left edge solid labels:   {left_labels}")
print(f"  Right edge solid labels:  {right_labels}")

# Periodic connectivity: with periodic BCs, top connects to bottom, left connects to right
# So we need to check: do any components "touch" across periodic boundaries?
# A Y-direction load path requires solid that crosses from top to bottom (either directly or via left-right PBC)
print()
print("Perimeter pixel connectivity analysis:")
print(f"  Top-Bottom same component? {bool(top_labels & bot_labels)}")
print(f"  Left-Right same component? {bool(left_labels & right_labels)}")

# ================================================================
# TEST 1: Homogenize the raw quarter-cell (as homogenize.py does)
# ================================================================
print("\n--- TEST 1: Homogenize quarter cell as-is (128x128) ---")
C_eff_quarter = homogenize_elastic(w, h, density_quarter)
kappa_eff_quarter = homogenize_thermal(w, h, density_quarter)
print(f"C11 = {C_eff_quarter[0,0]:.6e}")
print(f"C22 = {C_eff_quarter[1,1]:.6e}")
print(f"C12 = {C_eff_quarter[0,1]:.6e}")
print(f"C66 = {C_eff_quarter[2,2]:.6e}")
print(f"k11 = {kappa_eff_quarter[0,0]:.6e}")
print(f"k22 = {kappa_eff_quarter[1,1]:.6e}")
print(f"C11/C22 ratio = {C_eff_quarter[0,0]/C_eff_quarter[1,1]:.2e}")

# ================================================================
# TEST 2: Create full cell by 4-fold mirror symmetry
# ================================================================
# The correct reconstruction from a quarter-cell:
# Quarter is TOP-LEFT [0,1]x[0,1]
# Full cell is 2x2: TL=original, TR=mirror LR, BL=mirror UD, BR=mirror both
top_half = np.hstack([density_quarter, np.fliplr(density_quarter)])
full_cell = np.vstack([top_half, np.flipud(top_half)])
H, W = full_cell.shape
print(f"\n--- TEST 2: Homogenize full mirror-symmetric cell ({W}x{H}) ---")
print(f"Full cell solid fraction = {full_cell.mean():.4f}")

labeled_full, ncomp_full = label(full_cell)
top_f = set(labeled_full[0, :][labeled_full[0, :] > 0])
bot_f = set(labeled_full[H-1, :][labeled_full[H-1, :] > 0])
left_f = set(labeled_full[:, 0][labeled_full[:, 0] > 0])
right_f = set(labeled_full[:, W-1][labeled_full[:, W-1] > 0])
print(f"Full cell components: {ncomp_full}")
print(f"  Top-Bottom connected: {bool(top_f & bot_f)}")
print(f"  Left-Right connected: {bool(left_f & right_f)}")

C_eff_full = homogenize_elastic(W, H, full_cell)
kappa_eff_full = homogenize_thermal(W, H, full_cell)
print(f"C11 = {C_eff_full[0,0]:.6e}")
print(f"C22 = {C_eff_full[1,1]:.6e}")
print(f"C12 = {C_eff_full[0,1]:.6e}")
print(f"C66 = {C_eff_full[2,2]:.6e}")
print(f"k11 = {kappa_eff_full[0,0]:.6e}")
print(f"k22 = {kappa_eff_full[1,1]:.6e}")
print(f"C11/C22 ratio = {C_eff_full[0,0]/C_eff_full[1,1]:.2e}")

# ================================================================
# TEST 3: Compare - what SHOULD C22 be physically?
# ================================================================
print()
print("=" * 65)
print("COMPARISON:")
print(f"  Quarter cell: C11={C_eff_quarter[0,0]:.4e}, C22={C_eff_quarter[1,1]:.4e}")
print(f"  Full cell:    C11={C_eff_full[0,0]:.4e},    C22={C_eff_full[1,1]:.4e}")
print()
print("INTERPRETATION:")
if C_eff_full[1,1] > 100 * C_eff_quarter[1,1]:
    print("  -> CONFIRMED BUG: The full (mirrored) cell has MUCH higher C22!")
    print("     The quarter-cell was NOT correctly handled as periodic.")
    print("     homogenize.py needs to do 4-fold mirroring before computing.")
elif abs(C_eff_full[1,1] - C_eff_quarter[1,1]) < 1e-6 * C_eff_full[1,1]:
    print("  -> No significant difference. Quarter cell IS the correct input.")
else:
    print(f"  -> C22 differs by {C_eff_full[1,1]/C_eff_quarter[1,1]:.2f}x.")

# ================================================================
# Save comparison figure
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Quarter (left) vs Full Mirror Cell (right)", fontsize=13, weight='bold')
axes[0].imshow(density_quarter, cmap='gray_r', vmin=0, vmax=1)
axes[0].set_title("Quarter cell (current input)\n"
                   f"C11={C_eff_quarter[0,0]:.3e}, C22={C_eff_quarter[1,1]:.3e}\n"
                   f"k11={kappa_eff_quarter[0,0]:.3e}, k22={kappa_eff_quarter[1,1]:.3e}", fontsize=9)
axes[1].imshow(full_cell, cmap='gray_r', vmin=0, vmax=1)
axes[1].set_title("4-fold mirrored full cell (correct physical structure)\n"
                   f"C11={C_eff_full[0,0]:.3e}, C22={C_eff_full[1,1]:.3e}\n"
                   f"k11={kappa_eff_full[0,0]:.3e}, k22={kappa_eff_full[1,1]:.3e}", fontsize=9)
plt.tight_layout()
plt.savefig("Output/homogenize_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print()
print("Comparison saved to: Output/homogenize_comparison.png")
