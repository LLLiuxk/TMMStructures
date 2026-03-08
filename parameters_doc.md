# V2 Microstructure Generator - Parameter Space Documentation

This document lists all the adjustable parameters used by the `generate_microstructure.py` script. The generator takes a 1D vector of length **47**, with all values normalized to the range `[0.0, 1.0]`. You can sample from this space to generate a large dataset of periodic microstructures.

## Parameter Vector Breakdown (`params.shape = (47,)`)

### 1. Boundary Node Coordinates (Indices 0 - 11)
These parameters determine the placement of connection nodes on the boundaries. To ensure periodicity, nodes on opposite boundaries share the same coordinates.

*   **`params[0:3]`**: $X$-coordinates for the **Top and Bottom** boundaries. 
    *   3 nodes are positioned on the Top edge ($y=0$) and duplicated exactly on the Bottom edge ($y=255$).
    *   Mapping: `x_coordinate = param * 255`
*   *`params[3:6]`*: (Reserved / Unused in current symmetric mapping)
*   **`params[6:9]`**: $Y$-coordinates for the **Left and Right** boundaries.
    *   3 nodes are positioned on the Left edge ($x=0$) and duplicated exactly on the Right edge ($x=255$).
    *   Mapping: `y_coordinate = param * 255`
*   *`params[9:12]`*: (Reserved / Unused in current symmetric mapping)

### 2. Topology / Edge Connectivity (Indices 12 - 26)
There are exactly 6 node pairs (3 Top/Bottom, 3 Left/Right). Exploring all possible pairs yields $\frac{6 \times 5}{2} = 15$ potential edges.

*   **`params[12:27]`**: Edge activation weights (15 values).
    *   **Generation Rule**: The algorithm sorts these 15 weights in descending order. It uses a **Minimum Spanning Tree (MST)** approach (Kruskal's algorithm) to guarantee that all activated nodes are fully connected into a single solid graph (eliminating floating components).
    *   After forming the connected tree (which takes 5 edges), it adds up to **2 extra edges** (to create loops) provided their weight is $> 0.6$. This constraint naturally minimizes excessive crossings and overlaps.

### 3. Geometry & Morphological Curvature (Indices 27 - 41)
Rather than simple straight lines, the structure can bend utilizing Quadratic Bezier Curves.

*   **`params[27:42]`**: Lateral control point offsets for the 15 edges.
    *   Mapping: `offset = param * 2.0 - 1.0` (Mapped to `[-1.0, 1.0]`).
    *   If the underlying edge is activated, this parameter determines how much the strut bends away from a straight line. An offset of exactly 0.5 (mapped to 0.0) yields a perfectly straight strut.

### 4. Global Thickness (Index 42)
*   **`params[42]`**: Controls the global strut width.
    *   **Base Width**: Fixed at `20%` of the image size ($256 \times 0.2 \approx 51.2$ pixels) to ensure a significant modulus.
    *   **Mapping Rule**: `Thickness = round( Base_Width * (0.5 + param) )`
    *   This means the generated thickness smoothly varies from $0.5\times$ to $1.5\times$ of the base width (approx. 25 pixels to 76 pixels).

### 5. Reserved Padding (Indices 43 - 46)
*   **`params[43:47]`**: Currently not modifying the geometry (can be sampled randomly or set to zero for future use).


## Sampling Recommendation
When generating large-scale datasets for diffusion models:
1. Draw `params` from a uniform distribution `U(0, 1)` for all 47 dimensions.
2. Pass the array into `generator.generate(params, output_path)`.
3. The built-in physical rules (MST connectivity + thickness bounds) inherently guarantee that the resulting microstructure will be fully connected, perfectly periodic, and free from extremely thin, disconnected floating artifacts.
