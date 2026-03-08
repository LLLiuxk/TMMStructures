import numpy as np
from PIL import Image, ImageDraw
import os

class MicrostructureGeneratorV2:
    """
    Parametric generator for 2D periodic microstructures (V2: Boundary-Edge Point Topology).
    Generates full 256x256 images.
    
    Parameters structure (array of shape 47,):
    Indices 0-5:    Top/Bottom boundary node x-coordinates (3 pairs, mapped [0, 255])
    Indices 6-11:   Left/Right boundary node y-coordinates (3 pairs, mapped [0, 255])
    Indices 12-26:  Topology matrix (15 possible connections between the 6 distinct node pairs) in [0, 1] -> bool
    Indices 27-41:  Bezier curve lateral offsets (-1.0 to 1.0) for the 15 connections
    Index 42:       Global line thickness scaling in [0, 1]
    Index 43-46:    Reserved/Padding
    """
    def __init__(self, size=256):
        self.size = size
        
        # We define 6 'node groups' (pairs). A connection between group i and group j
        # means we must draw the exact same connection taking into account periodicity.
        # Group 0-2: Top/Bottom nodes (x is variable, y is 0 and 255)
        # Group 3-5: Left/Right nodes (x is 0 and 255, y is variable)
        
        # 15 connections between the 6 groups (n*(n-1)/2 = 6*5/2)
        self.all_group_edges = []
        for i in range(6):
            for j in range(i+1, 6):
                self.all_group_edges.append((i, j))
                
    def _get_cubic_bezier_points(self, p1, p2, offset, num_pts=20):
        """
        Calculates points for a bezier curve between p1 and p2, deflected by offset.
        offset is normal to the segment p1-p2.
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # Midpoint
        mx, my = (x1+x2)/2.0, (y1+y2)/2.0
        
        # Vector p1->p2
        dx, dy = x2-x1, y2-y1
        length = np.hypot(dx, dy)
        
        if length < 1e-5:
            return [p1, p2]
            
        # Normal vector
        nx, ny = -dy/length, dx/length
        
        # Deflect control point by offset * length
        cx = mx + nx * offset * length * 0.5
        cy = my + ny * offset * length * 0.5
        
        # Quadratic Bezier
        pts = []
        for t in np.linspace(0, 1, num_pts):
            xt = (1-t)**2 * x1 + 2*(1-t)*t*cx + t**2 * x2
            yt = (1-t)**2 * y1 + 2*(1-t)*t*cy + t**2 * y2
            pts.append((xt, yt))
        return pts
        
    def generate(self, params, output_path=None):
        params = np.array(params)
        
        # 1. Decode coordinates
        t_x = params[0:3] * (self.size - 1)
        l_y = params[6:9] * (self.size - 1)
        
        # Node instances: list of lists.
        # nodes[group_id] = [point_instance_1, point_instance_2]
        # For Top/Bottom groups (0, 1, 2)
        nodes = []
        for i in range(3):
            # Top point, Bottom point
            nodes.append([(t_x[i], 0), (t_x[i], self.size - 1)])
            
        # For Left/Right groups (3, 4, 5)
        for i in range(3):
            # Left point, Right point
            nodes.append([(0, l_y[i]), (self.size - 1, l_y[i])])
            
        # 2. Decode topology
        edge_weights = params[12:27]
        offsets = params[27:42] * 2.0 - 1.0  # map to [-1, 1]
        
        # --- Topology Rules ---
        # 1. Guarantee connectivity (Minimum Spanning Tree-like approach)
        # 2. Minimize excessive crossings (Sparsity constraints)
        
        edge_indices = np.argsort(edge_weights)[::-1]  # Sort descending
        parent = list(range(6))
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False
        
        edge_activations = np.zeros(15, dtype=bool)
        edges_added = 0
        
        # Step 1: Spanning tree for connectivity
        for idx in edge_indices:
            u, v = self.all_group_edges[idx]
            if union(u, v):
                edge_activations[idx] = True
                edges_added += 1
                if edges_added == 5: # 6 nodes need 5 edges to be fully connected
                    break
                    
        # Step 2: Add a few more edges if weight is high, but avoid too many 
        MAX_EXTRA_EDGES = 2 # Max extra edges to limit crossings
        extra_edges = 0
        for idx in edge_indices:
            if not edge_activations[idx] and edge_weights[idx] > 0.6:
                edge_activations[idx] = True
                extra_edges += 1
                if extra_edges >= MAX_EXTRA_EDGES:
                    break
        
        # --- Geometry Rules ---
        # Base thickness is 20% of boundary length 
        base_thickness = self.size * 0.2
        # Adjust based on parameter (range: 0.5x to 1.5x of base thickness)
        thickness = int(round(base_thickness * (0.5 + params[42])))
        
        img = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(img)
        
        active_instances = set()
        
        # 3. Draw edges with periodicity
        for k, (u, v) in enumerate(self.all_group_edges):
            if not edge_activations[k]:
                continue
                
            offset = offsets[k]
            
            # To maintain periodicity, if we connect group U to group V,
            # we must choose the instances (or wrap around) so that the line 
            # drawn physically matches when tiled.
            # Easiest way to handle straight/curved lines that cross boundaries:
            # We draw the connection from U_instance1 to V_instanceX, 
            # and replicate it at all periodic offsets (-self.size, 0, +self.size).
            
            # Let's just pick the primary instance of U and V (e.g. Top/Left)
            p1 = nodes[u][0]
            p2 = nodes[v][0]
            
            # We draw the line, but we also draw 9 translated copies to ensure 
            # boundary wrapping is perfectly captured in the 256x256 window.
            pts = self._get_cubic_bezier_points(p1, p2, offset, num_pts=30)
            
            for dx in [-self.size, 0, self.size]:
                for dy in [-self.size, 0, self.size]:
                    translated_pts = [(x+dx, y+dy) for x,y in pts]
                    # Draw curve segments
                    for idx in range(len(translated_pts)-1):
                        draw.line([translated_pts[idx], translated_pts[idx+1]], fill=0, width=thickness)
            
            active_instances.add(nodes[u][0])
            active_instances.add(nodes[u][1])
            active_instances.add(nodes[v][0])
            active_instances.add(nodes[v][1])
            
        # Draw joints
        r = thickness / 2.0
        for (cx, cy) in active_instances:
            # Draw joints at primary and translated positions
            for dx in [-self.size, 0, self.size]:
                for dy in [-self.size, 0, self.size]:
                    tx, ty = cx+dx, cy+dy
                    if -r <= tx <= self.size+r and -r <= ty <= self.size+r:
                        bbox = [int(round(tx - r)), int(round(ty - r)), 
                                int(round(tx + r)), int(round(ty + r))]
                        draw.ellipse(bbox, fill=0)
            
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            img.save(output_path)
            
        img_array = np.array(img, dtype=np.float64) / 255.0
        binary_array = (img_array < 0.5).astype(np.float64)
        return binary_array

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate parameterized microstructure PNGs (V2).")
    parser.add_argument('--sample', type=int, default=5, help='Number of random samples to generate')
    args = parser.parse_args()
    
    generator = MicrostructureGeneratorV2()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Input', 'figures')
    
    print(f"Generating {args.sample} random microstructures (V2)...")
    np.random.seed(42)
    
    for i in range(args.sample):
        params = np.random.rand(47)
        # Ensure some connectivity
        params[12:27] = np.random.uniform(0.1, 1.0, size=15)
        # Make some curves highly curved, others straight
        params[27:42] = np.random.uniform(0.0, 1.0, size=15)
        
        out_path = os.path.join(out_dir, f'sample_v2_{i+1:03d}.png')
        generator.generate(params, output_path=out_path)
        print(f"Saved: {out_path}")
