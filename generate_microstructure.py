"""
Generate Microstructure Samples from Parameters
=============================================
Reads a parameterization schema (nodes and connections)
and renders a 128x128 binary quarter-cell microstructure image.
"""

import numpy as np
import math
from PIL import Image, ImageDraw
import os

def lines_intersect(p1, p2, p3, p4):
    """
    Checks if line segment (p1, p2) intersects with (p3, p4).
    Uses standard CCW geometric check.
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def get_bezier_curve(p0, p1, p2, p3, steps=50):
    """
    Generates a list of points representing a cubic Bezier curve.
    """
    curve_pts = []
    for t in np.linspace(0, 1, steps):
        inv_t = 1.0 - t
        x = (inv_t**3 * p0[0] + 
             3 * inv_t**2 * t * p1[0] + 
             3 * inv_t * t**2 * p2[0] + 
             t**3 * p3[0])
        y = (inv_t**3 * p0[1] + 
             3 * inv_t**2 * t * p1[1] + 
             3 * inv_t * t**2 * p2[1] + 
             t**3 * p3[1])
        curve_pts.append((x, y))
    return curve_pts

def get_normal_vector(edge_id):
    """
    Returns the normal vector (dx, dy) pointing INWARD from the specified outer edge.
    E1 (Top) -> (0, 1)
    E2 (Right) -> (-1, 0)
    E3 (Bottom) -> (0, -1)
    E4 (Left) -> (1, 0)
    """
    if edge_id == "E1": return (0, 1)
    if edge_id == "E2": return (-1, 0)
    if edge_id == "E3": return (0, -1)
    if edge_id == "E4": return (1, 0)
    return (0, 0)

def render_microstructure(params, size=(128, 128), out_path="sample.png"):
    """
    Renders a 1/4 microstructure based on parameters.
    size: (width, height) in pixels
    """
    # Create white background (void = 255)
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    w, h = size
    
    def map_pt(pt):
        # pt in [0, 1]x[0,1]
        return (pt[0] * w, pt[1] * h)

    # 1. Parse Nodes
    points_map = {}
    nodes_def = params.get("nodes", {})
    
    for edge_id, arr in nodes_def.items():
        for i, val in enumerate(arr):
            pos = val[0]
            width_norm = val[1]
            
            p, p1, p2 = None, None, None
            if edge_id == "E1": # Top (y=0)
                p = (pos, 0.0)
                p1 = (pos - width_norm/2, 0.0)
                p2 = (pos + width_norm/2, 0.0)
            elif edge_id == "E2": # Right (x=1)
                p = (1.0, pos)
                p1 = (1.0, pos - width_norm/2)
                p2 = (1.0, pos + width_norm/2)
            elif edge_id == "E3": # Bottom (y=1)
                p = (pos, 1.0)
                p1 = (pos - width_norm/2, 1.0)
                p2 = (pos + width_norm/2, 1.0)
            elif edge_id == "E4": # Left (x=0)
                p = (0.0, pos)
                p1 = (0.0, pos - width_norm/2)
                p2 = (0.0, pos + width_norm/2)
                
            if p:
                points_map[(edge_id, i)] = {"edge": edge_id, "p": p, "p1": p1, "p2": p2}

    # 2. Draw Connections
    connections = params.get("connections", [])
    for conn in connections:
        p_start_ref = tuple(conn["start"])
        p_end_ref = tuple(conn["end"])
        c_type = conn.get("type", "straight_line")
        
        A = points_map[p_start_ref]
        B = points_map[p_end_ref]
        
        a1 = A["p1"]
        a2 = A["p2"]
        b1 = B["p1"]
        b2 = B["p2"]
        
        # Determine the non-intersecting mapping
        # We want to connect a1 -> b_x and a2 -> b_y
        if lines_intersect(a1, b1, a2, b2):
            left_end = b2
            right_end = b1
        else:
            left_end = b1
            right_end = b2

        if c_type == "straight_line":
            # Map down to pixels and draw the quad
            aq1 = map_pt(a1)
            aq2 = map_pt(a2)
            bq1 = map_pt(left_end)
            bq2 = map_pt(right_end)
            poly = [aq1, bq1, bq2, aq2]
            draw.polygon(poly, fill=0)
            
        elif c_type == "bezier_curve":
            # Compute bezier curves for both the left line (a1 -> left_end) 
            # and right line (a2 -> right_end).
            n_A = get_normal_vector(A["edge"])
            n_B = get_normal_vector(B["edge"])
            dist = math.hypot(A["p"][0] - B["p"][0], A["p"][1] - B["p"][1])
            push = dist * 0.4
            
            # Left Curve
            a1_c1 = (a1[0] + n_A[0]*push, a1[1] + n_A[1]*push)
            b1_c2 = (left_end[0] + n_B[0]*push, left_end[1] + n_B[1]*push)
            curve_left = get_bezier_curve(a1, a1_c1, b1_c2, left_end)
            
            # Right Curve
            a2_c1 = (a2[0] + n_A[0]*push, a2[1] + n_A[1]*push)
            b2_c2 = (right_end[0] + n_B[0]*push, right_end[1] + n_B[1]*push)
            curve_right = get_bezier_curve(a2, a2_c1, b2_c2, right_end)
            
            poly_points = [map_pt(pt) for pt in curve_left]
            poly_points.extend([map_pt(pt) for pt in reversed(curve_right)])
            draw.polygon(poly_points, fill=0)

        elif c_type == "tapered_line":
            # Tapered Line: Smooth log-like transition of thickness.
            # We can reuse the bezier logic, but the control points just stay exactly on the straight line
            # connecting the centers, while adjusting the width exponentially or smoothed.
            # A simple way to get a smooth bone-like taper is to use a bezier curve for the edges 
            # with control points hugging the straight line but "pinched" towards the thinner end or center.
            
            # Left Edge Bezier
            l_c1 = (a1[0] * 0.6 + left_end[0] * 0.4, a1[1] * 0.6 + left_end[1] * 0.4)
            l_c2 = (a1[0] * 0.4 + left_end[0] * 0.6, a1[1] * 0.4 + left_end[1] * 0.6)
            curve_left = get_bezier_curve(a1, l_c1, l_c2, left_end)
            
            # Right Edge Bezier
            r_c1 = (a2[0] * 0.6 + right_end[0] * 0.4, a2[1] * 0.6 + right_end[1] * 0.4)
            r_c2 = (a2[0] * 0.4 + right_end[0] * 0.6, a2[1] * 0.4 + right_end[1] * 0.6)
            curve_right = get_bezier_curve(a2, r_c1, r_c2, right_end)
            
            poly_points = [map_pt(pt) for pt in curve_left]
            poly_points.extend([map_pt(pt) for pt in reversed(curve_right)])
            draw.polygon(poly_points, fill=0)
            
        elif c_type == "circular_arc":
            # Circular Arc: Force a perfect circular curvature.
            # We approximate a circular arc using Bezier cubic curves. 
            # The magic constant for approximating a 90 degree circle arc is kappa = 0.552284749831
            # Here, we dynamically calculate the push based on the intersection of the border normals.
            n_A = get_normal_vector(A["edge"])
            n_B = get_normal_vector(B["edge"])
            
            # If the vectors are opposite (parallel), a circle arc is impossible (would just be a straight line or S curve)
            # fallback to straight_line logic
            if n_A[0] == -n_B[0] and n_A[1] == -n_B[1]:
                poly = [map_pt(a1), map_pt(left_end), map_pt(right_end), map_pt(a2)]
                draw.polygon(poly, fill=0)
            else:
                # Calculate intersection of normals to find "corner"
                push_A = abs(A["p"][0] - B["p"][0]) if n_A[0] != 0 else abs(A["p"][1] - B["p"][1])
                push_B = abs(A["p"][0] - B["p"][0]) if n_B[0] != 0 else abs(A["p"][1] - B["p"][1])
                
                kappa = 0.55228  # bezier circle approximation
                
                # Left Curve Arc
                a1_c1 = (a1[0] + n_A[0]*(push_A*kappa), a1[1] + n_A[1]*(push_A*kappa))
                b1_c2 = (left_end[0] + n_B[0]*(push_B*kappa), left_end[1] + n_B[1]*(push_B*kappa))
                curve_left = get_bezier_curve(a1, a1_c1, b1_c2, left_end)
                
                # Right Curve Arc
                a2_c1 = (a2[0] + n_A[0]*(push_A*kappa), a2[1] + n_A[1]*(push_A*kappa))
                b2_c2 = (right_end[0] + n_B[0]*(push_B*kappa), right_end[1] + n_B[1]*(push_B*kappa))
                curve_right = get_bezier_curve(a2, a2_c1, b2_c2, right_end)
                
                poly_points = [map_pt(pt) for pt in curve_left]
                poly_points.extend([map_pt(pt) for pt in reversed(curve_right)])
                draw.polygon(poly_points, fill=0)

    img.save(out_path)
    print(f"Saved {out_path}")

def main():
    os.makedirs('Output/sample_structures', exist_ok=True)
    
    # Sample 1: Straight lines (Cross truss)
    sample_1 = {
        "nodes": {
            "E1": [[0.5, 0.20]],
            "E2": [[0.5, 0.10]],
            "E3": [[0.5, 0.20]],
            "E4": [[0.5, 0.10]]
        },
        "connections": [
            {"start": ["E1", 0], "end": ["E4", 0], "type": "straight_line"},
            {"start": ["E2", 0], "end": ["E3", 0], "type": "straight_line"},
            {"start": ["E1", 0], "end": ["E2", 0], "type": "straight_line"}
        ]
    }
    
    # Sample 2: Tapered lines (Bone-like cross)
    sample_2 = {
        "nodes": {
            "E1": [[0.5, 0.30]],
            "E2": [[0.5, 0.10]],
            "E3": [[0.5, 0.30]],
            "E4": [[0.5, 0.10]]
        },
        "connections": [
            {"start": ["E1", 0], "end": ["E4", 0], "type": "tapered_line"},
            {"start": ["E2", 0], "end": ["E3", 0], "type": "tapered_line"},
            {"start": ["E1", 0], "end": ["E2", 0], "type": "tapered_line"}
        ]
    }

    # Sample 3: Circular Arcs loop 
    sample_3 = {
        "nodes": {
            "E1": [[0.5, 0.12]],
            "E2": [[0.5, 0.12]],
            "E3": [[0.5, 0.12]],
            "E4": [[0.5, 0.12]]
        },
        "connections": [
            {"start": ["E1", 0], "end": ["E2", 0], "type": "circular_arc"},
            {"start": ["E2", 0], "end": ["E3", 0], "type": "circular_arc"},
            {"start": ["E3", 0], "end": ["E4", 0], "type": "circular_arc"},
            {"start": ["E4", 0], "end": ["E1", 0], "type": "circular_arc"}
        ]
    }

    render_microstructure(sample_1, out_path="Output/sample_structures/generated_straight.png")
    render_microstructure(sample_2, out_path="Output/sample_structures/generated_tapered.png")
    render_microstructure(sample_3, out_path="Output/sample_structures/generated_arc.png")

if __name__ == "__main__":
    main()
