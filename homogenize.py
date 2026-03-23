"""
2D Microstructure Thermo-Mechanical Homogenization (Vectorized)
================================================================
Computes effective elastic stiffness (C_eff) and thermal conductivity (kappa_eff)
for 2D periodic microstructures using pixel-based FEM with periodic boundary conditions.

Input:  Binary PNG images representing the top-left 1/4 of the unit cell.
        Default: black=solid material, white=void (holes).
        Located in Input/figures/.
Output: Effective property matrices saved to Output/.

Method: Each pixel becomes a bilinear quadrilateral (Q4) finite element.
        Periodic boundary conditions are enforced via DOF reduction.
        Homogenization is performed by applying unit strain/gradient fields
        and computing volume-averaged stress/flux responses.

        Assembly is fully vectorized using numpy/scipy for performance.
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from PIL import Image
import os
import glob
import csv
import json
import time


# ==============================================================================
# 1. Image Loading & Symmetry Reconstruction
# ==============================================================================

def load_and_reconstruct(image_path, invert=True):
    """
    Load a quarter binary PNG image and reconstruct the full periodic unit cell
    via 4-fold mirror symmetry.
    
    Args:
        image_path: path to the PNG image (128x128 quarter cell)
        invert: if True (default), black pixels = solid, white pixels = void.
                if False, white pixels = solid, black pixels = void.

    Returns:
        density: 2D numpy array (256x256) with 1.0=solid, 0.0=void
    """
    img = Image.open(image_path).convert('L')
    quarter = np.array(img, dtype=np.float64) / 255.0

    if invert:
        # Black = solid (1.0), White = void (0.0)
        quarter = (quarter < 0.5).astype(np.float64)
    else:
        # White = solid (1.0), Black = void (0.0)
        quarter = (quarter > 0.5).astype(np.float64)

    # Reconstruct full 2x2 unit cell via mirror symmetry
    top_half = np.hstack((quarter, np.fliplr(quarter)))
    full = np.vstack((top_half, np.flipud(top_half)))

    return full


# ==============================================================================
# 2. Element Stiffness Matrices
# ==============================================================================

def element_stiffness_elastic(nu):
    """
    8x8 element stiffness matrix for a unit-sized Q4 element (plane stress, E=1).
    From Andreassen et al. 2011 "88-line code".
    """
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    KE = 1.0 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


def element_stiffness_thermal():
    """
    4x4 element conductivity matrix for a unit-sized Q4 element (k=1).
    """
    KE_th = 1.0 / 6.0 * np.array([
        [ 4, -1, -2, -1],
        [-1,  4, -1, -2],
        [-2, -1,  4, -1],
        [-1, -2, -1,  4]
    ])
    return KE_th


# ==============================================================================
# 3. B-matrix at Gauss points (precomputed)
# ==============================================================================

def precompute_B_matrices():
    """
    Precompute the B matrices (strain-displacement) at 2x2 Gauss points
    for a unit-sized Q4 element.

    Returns:
        B_all: list of 4 B matrices (3x8 each)
        B_th_all: list of 4 B_th matrices (2x4 each)
        gauss_wts: list of 4 weights
        detJ: Jacobian determinant (scalar, same for all points in unit element)
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    gauss_wts = [1.0, 1.0, 1.0, 1.0]

    B_all = []
    B_th_all = []
    detJ = 0.25  # For unit element

    for xi, eta in gauss_pts:
        dNdxi = np.array([
            [-(1-eta)/4,  (1-eta)/4,  (1+eta)/4, -(1+eta)/4],
            [-(1-xi)/4,  -(1+xi)/4,   (1+xi)/4,   (1-xi)/4]
        ])
        # For unit element: J = 0.5*I, inv(J) = 2*I
        dNdx = 2.0 * dNdxi

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]   = dNdx[0, i]
            B[1, 2*i+1] = dNdx[1, i]
            B[2, 2*i]   = dNdx[1, i]
            B[2, 2*i+1] = dNdx[0, i]
        B_all.append(B)

        B_th_all.append(dNdx)  # (2, 4)

    return B_all, B_th_all, gauss_wts, detJ


# ==============================================================================
# 4. Vectorized Global Assembly with Periodic BCs
# ==============================================================================

def build_periodic_node_map(nelx, nely):
    """
    Build the node-level periodic mapping.
    Maps right→left, bottom→top, corners→single node.

    Returns:
        node_map: array[n_nodes] mapping each node to its master
        compressed_map: array[n_nodes] mapping each node to compressed index
        n_free_nodes: number of independent nodes
    """
    n_nodes = (nelx + 1) * (nely + 1)
    node_master = np.arange(n_nodes)

    # Right boundary → Left boundary
    for iy in range(nely + 1):
        right = nelx * (nely + 1) + iy
        left = iy
        node_master[right] = left

    # Bottom boundary → Top boundary (follow chains)
    for ix in range(nelx + 1):
        bottom = ix * (nely + 1) + nely
        top = ix * (nely + 1)
        node_master[bottom] = node_master[top]

    unique_masters = np.unique(node_master)
    compressed_map = np.zeros(n_nodes, dtype=int)
    master_to_idx = {}
    for i, m in enumerate(unique_masters):
        master_to_idx[m] = i
    for node in range(n_nodes):
        compressed_map[node] = master_to_idx[node_master[node]]

    return compressed_map, len(unique_masters)


def build_element_dof_arrays(nelx, nely):
    """
    Vectorized construction of element DOF arrays.

    Returns:
        edof_elastic: (n_elem, 8) array of elastic DOFs per element
        edof_thermal: (n_elem, 4) array of thermal DOFs per element
    """
    # Element (ex, ey) nodes
    ey, ex = np.meshgrid(np.arange(nely), np.arange(nelx), indexing='ij')
    ey = ey.ravel()
    ex = ex.ravel()

    n1 = ex * (nely + 1) + ey
    n2 = (ex + 1) * (nely + 1) + ey
    n3 = (ex + 1) * (nely + 1) + (ey + 1)
    n4 = ex * (nely + 1) + (ey + 1)

    # Thermal DOFs = node indices
    edof_thermal = np.column_stack([n1, n2, n3, n4])  # (n_elem, 4)

    # Elastic DOFs = 2 per node
    edof_elastic = np.zeros((len(n1), 8), dtype=int)
    edof_elastic[:, 0] = 2 * n1
    edof_elastic[:, 1] = 2 * n1 + 1
    edof_elastic[:, 2] = 2 * n2
    edof_elastic[:, 3] = 2 * n2 + 1
    edof_elastic[:, 4] = 2 * n3
    edof_elastic[:, 5] = 2 * n3 + 1
    edof_elastic[:, 6] = 2 * n4
    edof_elastic[:, 7] = 2 * n4 + 1

    return edof_elastic, edof_thermal


def assemble_sparse(edof, KE_unit, elem_props, dof_map, n_free):
    """
    Vectorized sparse assembly using COO format.

    Args:
        edof: (n_elem, dof_per_elem) element DOF array (original numbering)
        KE_unit: (dof_per_elem, dof_per_elem) unit element matrix
        elem_props: (n_elem,) element property scaling factors
        dof_map: DOF mapping for periodic BCs
        n_free: number of free DOFs

    Returns:
        K: sparse (n_free x n_free) matrix
    """
    n_elem, ndof = edof.shape

    # Map to periodic DOFs
    edof_p = dof_map[edof]  # (n_elem, ndof)

    # Build COO triplets vectorized
    # For each element, we have ndof*ndof entries
    iK = np.repeat(edof_p, ndof, axis=1)  # (n_elem, ndof*ndof) - rows
    jK = np.tile(edof_p, (1, ndof))  # (n_elem, ndof*ndof) - cols

    # KE entries scaled by element property
    sK = np.outer(elem_props, KE_unit.ravel())  # (n_elem, ndof*ndof)

    K = coo_matrix((sK.ravel(), (iK.ravel(), jK.ravel())),
                   shape=(n_free, n_free)).tocsc()
    return K


def compute_elem_props(density_flat, prop_solid, prop_min, penal):
    """SIMP material interpolation."""
    return prop_min + density_flat**penal * (prop_solid - prop_min)


# ==============================================================================
# 5. Vectorized Load Vectors
# ==============================================================================

def assemble_elastic_loads(nelx, nely, density, E0, Emin, nu, penal,
                           edof_elastic, dof_map_elastic, n_free_elastic):
    """
    Vectorized computation of elastic load vectors for 3 unit strain fields.
    """
    B_all, _, gauss_wts, detJ = precompute_B_matrices()
    unit_strains = np.eye(3)
    n_elem = nelx * nely

    density_flat = density.ravel()  # row-major: (ey, ex) ordering
    Ee = compute_elem_props(density_flat, E0, Emin, penal)

    edof_p = dof_map_elastic[edof_elastic]  # (n_elem, 8)

    F = np.zeros((n_free_elastic, 3))

    # D matrix template (for E=1): scale by Ee per element
    D_unit = 1.0 / (1 - nu**2) * np.array([
        [1,   nu,  0],
        [nu,  1,   0],
        [0,   0,   (1-nu)/2]
    ])

    # For each Gauss point
    for gp_idx in range(4):
        B = B_all[gp_idx]  # (3, 8)
        w = gauss_wts[gp_idx]

        # Compute Bᵀ @ D_unit @ ε⁰_k for each load case (fixed for all elements)
        # fe_unit[k] = Bᵀ @ D_unit @ unit_strains[:, k] * detJ * w
        # This is (8,) for each load case
        fe_unit = np.zeros((8, 3))
        for lc in range(3):
            fe_unit[:, lc] = B.T @ D_unit @ unit_strains[:, lc] * detJ * w

        # Scale by element modulus: fe_elem = Ee[:, None] * fe_unit[None, :, :]
        # fe_elem: (n_elem, 8, 3)
        for lc in range(3):
            # fe for each element: (n_elem, 8) = Ee[:, None] * fe_unit[None, :, lc]
            fe_lc = Ee[:, None] * fe_unit[None, :, lc]  # (n_elem, 8)
            # Scatter-add into F
            np.add.at(F[:, lc], edof_p.ravel(), fe_lc.ravel())

    return F


def assemble_thermal_loads(nelx, nely, density, k0, kmin, penal,
                            edof_thermal, dof_map_thermal, n_free_thermal):
    """
    Vectorized computation of thermal load vectors for 2 unit gradient fields.
    """
    _, B_th_all, gauss_wts, detJ = precompute_B_matrices()
    unit_grads = np.eye(2)
    n_elem = nelx * nely

    density_flat = density.ravel()
    ke = compute_elem_props(density_flat, k0, kmin, penal)

    edof_p = dof_map_thermal[edof_thermal]

    F_th = np.zeros((n_free_thermal, 2))

    for gp_idx in range(4):
        B_th = B_th_all[gp_idx]  # (2, 4)
        w = gauss_wts[gp_idx]

        fe_unit = np.zeros((4, 2))
        for lc in range(2):
            fe_unit[:, lc] = B_th.T @ unit_grads[:, lc] * detJ * w

        for lc in range(2):
            fe_lc = ke[:, None] * fe_unit[None, :, lc]
            np.add.at(F_th[:, lc], edof_p.ravel(), fe_lc.ravel())

    return F_th


# ==============================================================================
# 6. Homogenization Solvers
# ==============================================================================

def homogenize_elastic(nelx, nely, density, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0):
    """
    Compute effective elastic stiffness matrix C_eff (3x3, Voigt notation).

    Returns:
        C_eff: 3x3 effective stiffness matrix [σxx,σyy,σxy] = C_eff [εxx,εyy,γxy]
    """
    # Periodic DOF map (elastic: 2 DOFs per node)
    compressed_map, n_free_nodes = build_periodic_node_map(nelx, nely)
    n_free = 2 * n_free_nodes
    # Build elastic DOF map from node map
    n_nodes = (nelx + 1) * (nely + 1)
    dof_map = np.zeros(2 * n_nodes, dtype=int)
    dof_map[0::2] = 2 * compressed_map
    dof_map[1::2] = 2 * compressed_map + 1

    # Element DOF arrays
    edof_elastic, _ = build_element_dof_arrays(nelx, nely)

    # Element properties
    density_flat = density.ravel()
    Ee = compute_elem_props(density_flat, E0, Emin, penal)

    # Assemble global stiffness
    KE = element_stiffness_elastic(nu)
    K = assemble_sparse(edof_elastic, KE, Ee, dof_map, n_free)

    # Assemble load vectors
    F = assemble_elastic_loads(nelx, nely, density, E0, Emin, nu, penal,
                               edof_elastic, dof_map, n_free)

    # Fix DOFs to remove rigid body modes
    K_lil = K.tolil()
    F_mod = F.copy()
    for dof in [0, 1]:
        K_lil[dof, :] = 0
        K_lil[:, dof] = 0
        K_lil[dof, dof] = 1.0
        F_mod[dof, :] = 0
    K = K_lil.tocsc()

    # Solve
    chi = np.zeros((n_free, 3))
    for lc in range(3):
        chi[:, lc] = spsolve(K, F_mod[:, lc])

    # Compute C_eff via energy method (vectorized over elements)
    B_all, _, gauss_wts, detJ = precompute_B_matrices()
    unit_strains = np.eye(3)
    vol = nelx * nely

    edof_p = dof_map[edof_elastic]  # (n_elem, 8)

    C_eff = np.zeros((3, 3))

    D_unit = 1.0 / (1 - nu**2) * np.array([
        [1,   nu,  0],
        [nu,  1,   0],
        [0,   0,   (1-nu)/2]
    ])

    for gp_idx in range(4):
        B = B_all[gp_idx]  # (3, 8)
        w = gauss_wts[gp_idx]

        for i_lc in range(3):
            # chi values at element DOFs for load case i: (n_elem, 8)
            chi_i = chi[edof_p, i_lc]  # (n_elem, 8)
            # Total strain = ε⁰_i - B·χ_i for each element
            # B @ chi_i.T → (3, n_elem); then ε⁰_i[:, None] - result
            B_chi_i = (B @ chi_i.T).T  # (n_elem, 3)
            eps_i = unit_strains[:, i_lc][None, :] - B_chi_i  # (n_elem, 3)
            # Stress: σ_i = Ee[:, None] * D_unit @ eps_i.T
            stress_i = Ee[:, None] * (eps_i @ D_unit.T)  # (n_elem, 3)

            for j_lc in range(3):
                chi_j = chi[edof_p, j_lc]
                B_chi_j = (B @ chi_j.T).T
                eps_j = unit_strains[:, j_lc][None, :] - B_chi_j

                # C_eff_ij += sum_e (σ_i · ε_j) * detJ * w
                C_eff[i_lc, j_lc] += np.sum(stress_i * eps_j) * detJ * w

    C_eff /= vol
    C_eff = 0.5 * (C_eff + C_eff.T)
    return C_eff


def homogenize_thermal(nelx, nely, density, k0=1.0, kmin=1e-9, penal=3.0):
    """
    Compute effective thermal conductivity tensor kappa_eff (2x2).

    Returns:
        kappa_eff: 2x2 effective thermal conductivity tensor
    """
    compressed_map, n_free = build_periodic_node_map(nelx, nely)
    dof_map = compressed_map  # 1 DOF per node

    _, edof_thermal = build_element_dof_arrays(nelx, nely)

    density_flat = density.ravel()
    ke = compute_elem_props(density_flat, k0, kmin, penal)

    KE_th = element_stiffness_thermal()
    K_th = assemble_sparse(edof_thermal, KE_th, ke, dof_map, n_free)

    F_th = assemble_thermal_loads(nelx, nely, density, k0, kmin, penal,
                                   edof_thermal, dof_map, n_free)

    K_lil = K_th.tolil()
    F_mod = F_th.copy()
    K_lil[0, :] = 0
    K_lil[:, 0] = 0
    K_lil[0, 0] = 1.0
    F_mod[0, :] = 0
    K_th = K_lil.tocsc()

    chi_th = np.zeros((n_free, 2))
    for lc in range(2):
        chi_th[:, lc] = spsolve(K_th, F_mod[:, lc])

    # Compute kappa_eff
    _, B_th_all, gauss_wts, detJ = precompute_B_matrices()
    unit_grads = np.eye(2)
    vol = nelx * nely

    edof_p = dof_map[edof_thermal]
    kappa_eff = np.zeros((2, 2))

    for gp_idx in range(4):
        B_th = B_th_all[gp_idx]  # (2, 4)
        w = gauss_wts[gp_idx]

        for i_lc in range(2):
            chi_i = chi_th[edof_p, i_lc]  # (n_elem, 4)
            B_chi_i = (B_th @ chi_i.T).T  # (n_elem, 2)
            grad_i = unit_grads[:, i_lc][None, :] - B_chi_i
            flux_i = ke[:, None] * grad_i

            for j_lc in range(2):
                chi_j = chi_th[edof_p, j_lc]
                B_chi_j = (B_th @ chi_j.T).T
                grad_j = unit_grads[:, j_lc][None, :] - B_chi_j
                kappa_eff[i_lc, j_lc] += np.sum(flux_i * grad_j) * detJ * w

    kappa_eff /= vol
    kappa_eff = 0.5 * (kappa_eff + kappa_eff.T)
    return kappa_eff


# ==============================================================================
# 7. Verification Tests
# ==============================================================================

def test_solid_block(size=10):
    """Test: fully solid block should recover input material properties."""
    print("=" * 60)
    print("TEST: Solid block (should recover material properties)")
    print("=" * 60)

    density = np.ones((size, size))
    E0, nu, k0 = 1.0, 0.3, 1.0

    C_eff = homogenize_elastic(size, size, density, E0=E0, nu=nu)
    kappa_eff = homogenize_thermal(size, size, density, k0=k0)

    D_expected = E0 / (1 - nu**2) * np.array([
        [1,   nu,  0],
        [nu,  1,   0],
        [0,   0,   (1-nu)/2]
    ])

    print(f"\nC_eff (computed):\n{C_eff}")
    print(f"\nD (expected):\n{D_expected}")
    print(f"\nMax error in C_eff: {np.max(np.abs(C_eff - D_expected)):.6e}")

    print(f"\nkappa_eff (computed):\n{kappa_eff}")
    print(f"kappa_eff (expected): [[{k0}, 0], [0, {k0}]]")
    print(f"Max error in kappa_eff: {np.max(np.abs(kappa_eff - k0*np.eye(2))):.6e}")

    passed = (np.max(np.abs(C_eff - D_expected)) < 1e-6 and
              np.max(np.abs(kappa_eff - k0*np.eye(2))) < 1e-6)
    print(f"\n{'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_void_block(size=10):
    """Test: fully void block should give near-zero properties."""
    print("\n" + "=" * 60)
    print("TEST: Void block (should give near-zero properties)")
    print("=" * 60)

    density = np.zeros((size, size))

    C_eff = homogenize_elastic(size, size, density)
    kappa_eff = homogenize_thermal(size, size, density)

    print(f"\nC_eff (computed):\n{C_eff}")
    print(f"Max |C_eff|: {np.max(np.abs(C_eff)):.6e}")

    print(f"\nkappa_eff (computed):\n{kappa_eff}")
    print(f"Max |kappa_eff|: {np.max(np.abs(kappa_eff)):.6e}")

    passed = np.max(np.abs(C_eff)) < 1e-6 and np.max(np.abs(kappa_eff)) < 1e-6
    print(f"\n{'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_symmetry(size=20):
    """Test: C_eff should be symmetric; checkerboard → C11≈C22, κ11≈κ22."""
    print("\n" + "=" * 60)
    print("TEST: Symmetry (checkerboard pattern)")
    print("=" * 60)

    density = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                density[i, j] = 1.0

    C_eff = homogenize_elastic(size, size, density)
    kappa_eff = homogenize_thermal(size, size, density)

    print(f"\nC_eff:\n{C_eff}")
    print(f"Symmetry error: {np.max(np.abs(C_eff - C_eff.T)):.6e}")
    print(f"C11-C22 difference: {abs(C_eff[0,0] - C_eff[1,1]):.6e}")

    print(f"\nkappa_eff:\n{kappa_eff}")
    print(f"κ11-κ22 difference: {abs(kappa_eff[0,0] - kappa_eff[1,1]):.6e}")

    sym_ok = np.max(np.abs(C_eff - C_eff.T)) < 1e-10
    isotropy_ok = abs(C_eff[0,0] - C_eff[1,1]) < 1e-6
    passed = sym_ok and isotropy_ok
    print(f"\n{'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ==============================================================================
# 8. Main Entry Point
# ==============================================================================

def process_image(image_path, E0=1.0, nu=0.3, k0=1.0, Emin=1e-9, kmin=1e-9, penal=3.0, silent=False):
    """
    Process a single microstructure image and compute effective properties.
    """
    if not silent:
        name = os.path.basename(image_path)
        print(f"  Homogenizing: {name}")

    density = load_and_reconstruct(image_path)
    nely, nelx = density.shape
    vf = np.mean(density)

    C_eff = homogenize_elastic(nelx, nely, density, E0, Emin, nu, penal)
    kappa_eff = homogenize_thermal(nelx, nely, density, k0, kmin, penal)

    return {
        'nelx': nelx,
        'nely': nely,
        'volume_fraction': float(vf),
        'C_eff': C_eff.tolist(),
        'kappa_eff': kappa_eff.tolist(),
        'C11': float(C_eff[0, 0]), 'C12': float(C_eff[0, 1]), 'C22': float(C_eff[1, 1]),
        'C66': float(C_eff[2, 2]), 'C16': float(C_eff[0, 2]), 'C26': float(C_eff[1, 2]),
        'k11': float(kappa_eff[0, 0]), 'k12': float(kappa_eff[0, 1]), 'k22': float(kappa_eff[1, 1]),
    }


def main():
    """Main entry point: process all images in Input/figures/."""
    print("=" * 60)
    print("2D Microstructure Thermo-Mechanical Homogenization")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'Input', 'figures')
    output_dir = os.path.join(script_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        print("Running verification tests instead...")
        test_solid_block()
        test_void_block()
        test_symmetry()
        return

    print(f"Found {len(image_files)} image(s) in {input_dir}")

    E0 = 1.0
    nu = 0.3
    k0 = 1.0
    Emin = 1e-9
    kmin = 1e-9
    penal = 3.0

    results = []
    for img_path in image_files:
        result = process_image(img_path, E0, nu, k0, Emin, kmin, penal)
        results.append(result)

    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'nelx', 'nely', 'volume_fraction',
                         'C11', 'C12', 'C22', 'C66', 'C16', 'C26',
                         'k11', 'k12', 'k22'])
        for r in results:
            writer.writerow([
                r['name'], r['nelx'], r['nely'], f"{r['volume_fraction']:.6f}",
                f"{r['C11']:.6e}", f"{r['C12']:.6e}", f"{r['C22']:.6e}",
                f"{r['C66']:.6e}", f"{r['C16']:.6e}", f"{r['C26']:.6e}",
                f"{r['k11']:.6e}", f"{r['k12']:.6e}", f"{r['k22']:.6e}"
            ])
    print(f"\nResults saved to {csv_path}")

    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {json_path}")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running verification tests...\n")
        t1 = test_solid_block()
        t2 = test_void_block()
        t3 = test_symmetry()
        print(f"\n{'='*60}")
        print(f"Results: {'All PASSED ✓' if all([t1,t2,t3]) else 'Some FAILED ✗'}")
    else:
        main()
