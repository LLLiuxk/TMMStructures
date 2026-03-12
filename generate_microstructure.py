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
        
        if A["edge"] == B["edge"]:
            # Same edge connection: prevent crossover mathematically 
            # by sorting based on the relevant coordinate
            idx = 0 if A["edge"] in ["E1", "E3"] else 1 # x for Top/Bottom, y for Right/Left
            
            # Extract coordinates for sorting
            pts = [
                (a1[idx], a1), (a2[idx], a2), 
                (b1[idx], b1), (b2[idx], b2)
            ]
            pts.sort(key=lambda item: item[0])
            
            # The middle two points belong to the inner connection
            # The outer two points belong to the outer connection
            # We must map a1 to the correct b and a2 to the correct b
            
            # If A is to the left/top of B
            if min(a1[idx], a2[idx]) < min(b1[idx], b2[idx]):
                # A's rightmost point (a2 usually) connects to B's leftmost point
                left_end = b1 if b1[idx] < b2[idx] else b2
                right_end = b2 if b1[idx] < b2[idx] else b1
            else:
                left_end = b2 if b1[idx] > b2[idx] else b1
                right_end = b1 if b1[idx] > b2[idx] else b2

            # Re-assign mapping based on geometric order to strictly avoid crossover
            if a1[idx] < a2[idx]:
                left_mod, right_mod = (b2, b1) if b2[idx] > b1[idx] else (b1, b2)
                # Outer to outer, inner to inner
                left_end, right_end = left_mod, right_mod
            else:
                left_mod, right_mod = (b1, b2) if b1[idx] < b2[idx] else (b2, b1)
                left_end, right_end = left_mod, right_mod
                
        else:
            if lines_intersect(a1, b1, a2, b2):
                left_end = b2
                right_end = b1
            else:
                left_end = b1
                right_end = b2

        # Handle same-edge connection fallback before rendering blocks
        if A["edge"] == B["edge"] and c_type in ("straight_line", "tapered_line"):
            c_type = "bezier_curve"

        if c_type == "straight_line":
            # Map down to pixels and draw the quad
            aq1 = map_pt(a1)
            aq2 = map_pt(a2)
            bq1 = map_pt(left_end)
            bq2 = map_pt(right_end)
            poly = [aq1, bq1, bq2, aq2]
            draw.polygon(poly, fill=0)
        if c_type == "bezier_curve":
            # Compute bezier curves for both the left line (a1 -> left_end) 
            # and right line (a2 -> right_end).
            n_A = get_normal_vector(A["edge"])
            n_B = get_normal_vector(B["edge"])
            dist = math.hypot(A["p"][0] - B["p"][0], A["p"][1] - B["p"][1])
            
            if A["edge"] == B["edge"]:
                # Same edge connection: Make a structural U-turn / smooth half-bezier
                dist_outer = math.hypot(a1[0] - left_end[0], a1[1] - left_end[1])
                dist_inner = math.hypot(a2[0] - right_end[0], a2[1] - right_end[1])
                
                # To maintain exactly constant thickness throughout a 180 degree Bezier bend,
                # the outer curve MUST travel further outward than the inner curve.
                # Specifically, push_outer - push_inner must equal the line thickness!
                # Setting push = dist/2 is a good approximation for a half circle.
                # Left Curve (Outer)
                push_outer = dist_outer * 0.55
                a1_c1 = (a1[0] + n_A[0]*push_outer, a1[1] + n_A[1]*push_outer)
                b1_c2 = (left_end[0] + n_B[0]*push_outer, left_end[1] + n_B[1]*push_outer)
                curve_left = get_bezier_curve(a1, a1_c1, b1_c2, left_end)
                
                # Right Curve (Inner)
                push_inner = dist_inner * 0.55
                a2_c1 = (a2[0] + n_A[0]*push_inner, a2[1] + n_A[1]*push_inner)
                b2_c2 = (right_end[0] + n_B[0]*push_inner, right_end[1] + n_B[1]*push_inner)
                curve_right = get_bezier_curve(a2, a2_c1, b2_c2, right_end)
            else:
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
            # For a tapered line, we want the edges to bow inward.
            # Using bezier curves where the control points are pulled towards the centerline.
            center_start = A["p"]
            center_end = B["p"]
            
            mid_center = ((center_start[0] + center_end[0])/2, (center_start[1] + center_end[1])/2)
            
            # Pull control points towards the midpoint of the center line (50% pinch factor)
            pinch = 0.5
            l_c1 = (a1[0]*(1-pinch) + mid_center[0]*pinch, a1[1]*(1-pinch) + mid_center[1]*pinch)
            l_c2 = (left_end[0]*(1-pinch) + mid_center[0]*pinch, left_end[1]*(1-pinch) + mid_center[1]*pinch)
            curve_left = get_bezier_curve(a1, l_c1, l_c2, left_end)
            
            r_c1 = (a2[0]*(1-pinch) + mid_center[0]*pinch, a2[1]*(1-pinch) + mid_center[1]*pinch)
            r_c2 = (right_end[0]*(1-pinch) + mid_center[0]*pinch, right_end[1]*(1-pinch) + mid_center[1]*pinch)
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
            elif A["edge"] == B["edge"]:
                # Same edge connection: Make a structural U-turn / half-circle
                dist_A = math.hypot(a1[0] - left_end[0], a1[1] - left_end[1])
                dist_B = math.hypot(a2[0] - right_end[0], a2[1] - right_end[1])
                
                # For a 180 degree semi-circle, kappa is approx 1.333
                kappa = 1.3333
                
                push_A_outer = dist_A / 2
                push_B_inner = dist_B / 2
                
                # Left Curve Arc (Outer)
                a1_c1 = (a1[0] + n_A[0]*(push_A_outer*kappa), a1[1] + n_A[1]*(push_A_outer*kappa))
                b1_c2 = (left_end[0] + n_B[0]*(push_A_outer*kappa), left_end[1] + n_B[1]*(push_A_outer*kappa))
                curve_left = get_bezier_curve(a1, a1_c1, b1_c2, left_end)
                
                # Right Curve Arc (Inner)
                a2_c1 = (a2[0] + n_A[0]*(push_B_inner*kappa), a2[1] + n_A[1]*(push_B_inner*kappa))
                b2_c2 = (right_end[0] + n_B[0]*(push_B_inner*kappa), right_end[1] + n_B[1]*(push_B_inner*kappa))
                curve_right = get_bezier_curve(a2, a2_c1, b2_c2, right_end)
                
                poly_points = [map_pt(pt) for pt in curve_left]
                poly_points.extend([map_pt(pt) for pt in reversed(curve_right)])
                draw.polygon(poly_points, fill=0)
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
