"""
Verification: Does the homogenization code treat the image correctly?
=======================================================================
Checks 3 things:
1. What does sample_0000.png look like (the raw quarter)
2. What does the generator doc say about E1/E2/E3/E4 convention
3. Does load_and_reconstruct do any mirroring?
4. Reconstruct what the full periodic unit cell looks like
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

print("=" * 65)
print("  VERIFICATION: Homogenization Quarter Cell vs Full Cell")
print("=" * 65)

# ========================================================
# 1. Node convention from generate_microstructure.py
# ========================================================
print()
print("[1] Edge definitions (from generate_microstructure.py):")
print("      E1 = TOP edge    (y=0)")
print("      E2 = RIGHT edge  (x=1)")
print("      E3 = BOTTOM edge (y=1)")
print("      E4 = LEFT edge   (x=0)")
print()
print("   sample_0000 nodes:")
print("      E1: pos=0.12  on TOP    -> coord (0.12, 0.0)")
print("      E2: pos=0.2116 on RIGHT -> coord (1.0,  0.2116)")
print("      E3: pos=0.5724 on BOTTOM-> coord (0.5724, 1.0)")
print("      E4: pos=0.2749 on LEFT  -> coord (0.0,  0.2749)")
print()
print("   Connections:")
print("      E1->E2 (straight): TOP(0.12,0) -> RIGHT(1, 0.2116)  <top-right diagonal>")
print("      E1->E3 (straight): TOP(0.12,0) -> BOTTOM(0.5724,1) <VERTICAL rod!>")
print("      E2->E3 (bezier):   RIGHT(1,0.2116) -> BOTTOM(0.5724,1)")
print("      E3->E4 (circular_arc): BOTTOM(0.5724,1) -> LEFT(0,0.2749)")
print()

# ========================================================
# 2. Load actual image and check
# ========================================================
img_path = "Output/dataset/batch_1/images/sample_0000.png"
img = Image.open(img_path).convert('L')
density_raw = np.array(img, dtype=np.float64) / 255.0
density_full_raw = (density_raw < 0.5).astype(np.float64)
print(f"[2] Image size: {img.size}  (W x H)")
print(f"    Solid fraction (raw image): {density_full_raw.mean():.4f}")
print()

# ========================================================
# 3. Check what homogenize.py does
# ========================================================
print("[3] load_and_reconstruct in homogenize.py:")
print("    Comment: 'Load a full binary PNG image representing the complete unit cell.'")
print("    Comment: '(No longer mirrors 1/4 unit cells as per V2 generator).'")
print("    => The homogenization treats the 128x128 image AS-IS as the FULL unit cell.")
print("    => NO mirroring is done.")
print()

# ========================================================
# 4. Reconstruct what the full periodic unit cell SHOULD be
#    if the 128x128 image is treated as 1/4 (top-left)
# ========================================================
h, w = density_full_raw.shape

# Mirror to create full unit cell from quarter
# Quarter is top-left -> mirror horizontally first, then vertically
def make_full_from_quarter(q):
    top = np.hstack([q, np.fliplr(q)])         # top half (left + right mirror)
    full = np.vstack([top, np.flipud(top)])     # combine top and bottom mirror
    return full

density_reconstructed = make_full_from_quarter(density_full_raw)
print(f"[4] If quarter-cell mirroring were applied:")
print(f"    Full cell size: {density_reconstructed.shape}")
print(f"    Solid fraction (mirrored): {density_reconstructed.mean():.4f}")
print()

# ========================================================
# 5. Visual comparison
# ========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("sample_0000 Homogenization Verification", fontsize=14, weight='bold')

# Panel 1: Raw image (as homogenize.py sees it)
ax1 = axes[0]
ax1.imshow(density_full_raw, cmap='gray_r', vmin=0, vmax=1, origin='upper')
ax1.set_title("What homogenize.py receives\n(128x128, treated as FULL unit cell)", fontsize=10)
ax1.set_xlabel("x"); ax1.set_ylabel("y")
# Mark edges
for spine in ax1.spines.values():
    spine.set_edgecolor('red'); spine.set_linewidth(3)
ax1.text(5, 10, "E1\n(top)", color='blue', fontsize=8)
ax1.text(110, 64, "E2\n(right)", color='green', fontsize=7)
ax1.text(64, 120, "E3\n(bottom)", color='orange', fontsize=8)
ax1.text(2, 64, "E4\n(left)", color='purple', fontsize=8)

# Mark nodes
nodes = {
    'E1(top,0.12)':   (0.12 * 128, 0),
    'E2(right,0.21)': (128,       0.2116 * 128),
    'E3(bot,0.57)':   (0.5724*128, 128),
    'E4(left,0.27)':  (0,          0.2749*128),
}
for name, (nx, ny) in nodes.items():
    ax1.plot(nx, ny, 'ro', markersize=8, zorder=5)

# Panel 2: CORRECT full unit cell if mirroring applied
ax2 = axes[1]
ax2.imshow(density_reconstructed, cmap='gray_r', vmin=0, vmax=1, origin='upper')
ax2.set_title("What it SHOULD be\n(256x256, after 4-fold mirror symmetry)", fontsize=10)
ax2.set_xlabel("x"); ax2.set_ylabel("y")
# Draw quarter boundaries
ax2.axvline(x=128, color='red', lw=1, linestyle='--', alpha=0.7)
ax2.axhline(y=128, color='red', lw=1, linestyle='--', alpha=0.7)
ax2.text(5, 10, "TL\n(original)", color='red', fontsize=7)
ax2.text(135, 10, "TR\n(mirror)", color='red', fontsize=7)
ax2.text(5, 135, "BL\n(mirror)", color='red', fontsize=7)
ax2.text(135, 135, "BR\n(mirror)", color='red', fontsize=7)

# Panel 3: Explanation diagram
ax3 = axes[2]
ax3.set_xlim(0, 2); ax3.set_ylim(2, 0)
ax3.set_aspect('equal')
ax3.set_title("Coordinate conventions\n(unit cell)", fontsize=10)

# Draw unit square
rect = plt.Rectangle((0,0),1,1, fill=False, ec='black', lw=2)
ax3.add_patch(rect)

# Draw connections for sample_0000
E1 = (0.12, 0.0)   # top
E2 = (1.0, 0.2116) # right
E3 = (0.5724, 1.0) # bottom
E4 = (0.0, 0.2749) # left

for pt, name, col in [(E1,'E1-top','blue'), (E2,'E2-right','green'), 
                       (E3,'E3-bottom','orange'), (E4,'E4-left','purple')]:
    ax3.plot(pt[0], pt[1], 'o', markersize=10, color=col)
    ax3.annotate(name, pt, xytext=(pt[0]+0.05, pt[1]+0.05), fontsize=8, color=col)

ax3.plot([E1[0], E2[0]], [E1[1], E2[1]], 'b-', lw=2, label='E1-E2 straight')
ax3.plot([E1[0], E3[0]], [E1[1], E3[1]], 'r-', lw=3, label='E1-E3 VERTICAL', alpha=0.9)
ax3.plot([E2[0], E3[0]], [E2[1], E3[1]], 'g--', lw=2, label='E2-E3 bezier')
ax3.plot([E3[0], E4[0]], [E3[1], E4[1]], 'm--', lw=2, label='E3-E4 arc')

ax3.text(0.3, 0.5, 'E1->E3\n(vertical\nrod!)', color='red', fontsize=10, weight='bold',
         ha='center', va='center')

ax3.set_xlabel("x (0=left, 1=right)")
ax3.set_ylabel("y (0=top, 1=bottom)")
ax3.legend(loc='lower right', fontsize=7)
ax3.set_title("sample_0000 in [0,1]^2 coords\n(y: 0=top, 1=bottom)", fontsize=9)

plt.tight_layout()
out = "Output/verify_homogenize.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()

print(f"[5] Verification figure saved to: {out}")
print()

# ========================================================
# 6. Critical question: if the E1->E3 rod spans top to bottom,
#    does that provide Y-direction load path?
# ========================================================
print("[6] CRITICAL ANALYSIS:")
print()
print("    The E1->E3 connection (straight_line) goes from:")
print("      E1: (0.12, 0.0) [top edge]")
print("      E3: (0.5724, 1.0) [bottom edge]")
print()
print("    This is a DIAGONAL rod (not vertical), going from top-left to bottom-right.")
print()
print("    When the 128x128 image is treated as the FULL periodic unit cell,")
print("    the periodic boundary conditions link:")
print("      TOP <-> BOTTOM  (Y-periodicity)")
print("      LEFT <-> RIGHT  (X-periodicity)")
print()
print("    The diagonal rod DOES cross from top to bottom -> Y-periodicity path EXISTS.")
print()
print("    BUT: If the image was supposed to be a QUARTER CELL that needs mirroring,")
print("    the mirrored full cell would have a very different topology,")
print("    and the E1->E3 rod would appear at DIFFERENT positions in the full cell.")
print()
print("    ==> KEY QUESTION: Is the 128x128 image a FULL cell or a QUARTER cell?")
print()

# Check what fraction of the image is black (solid) near the top-bottom path
col_solid = [density_full_raw[:, int(col)].sum() for col in range(128)]
top_row = density_full_raw[0, :].sum()
bot_row = density_full_raw[127, :].sum()
print(f"    Solid pixels in top row: {top_row:.0f} / 128")
print(f"    Solid pixels in bot row: {bot_row:.0f} / 128")

# Check connectivity: is there a continuous solid path from top to bottom?
# Simple flood fill check
from scipy.ndimage import label
labeled, num_features = label(density_full_raw)
print(f"    Number of connected solid regions: {num_features}")

# Check if top-row and bottom-row belong to same component
if num_features > 0:
    top_labels = set(labeled[0, :][labeled[0, :] > 0])
    bot_labels = set(labeled[127, :][labeled[127, :] > 0])
    shared = top_labels & bot_labels
    print(f"    Solid labels at top edge: {top_labels}")
    print(f"    Solid labels at bot edge: {bot_labels}")
    print(f"    Shared (top-to-bottom connected): {shared}")
    if shared:
        print("    => YES: there is a solid path from top to bottom edge!")
        print("       But homogenization uses PERIODIC BCs, not fixed boundary conditions.")
        print("       With periodic BCs, the FEM solves for the periodic displacement field,")
        print("       so even a partial path contributes to Y-stiffness.")
    else:
        print("    => NO: top and bottom are NOT connected in the raw image!")
        print("       This confirms C22 ~ Emin (Y-direction has no load bearing path).")

print()
print("=" * 65)
