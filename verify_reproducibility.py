import os
import json
import numpy as np
from generate_microstructure import render_microstructure
from homogenize import process_image
from plot_combined_radar import save_radar_chart

def verify_sample(sample_input, dataset_path="Output/dataset/batch_1"):
    # Allow input as simple number (e.g., 42 -> sample_0042)
    if sample_input.isdigit():
        sample_id = f"sample_{int(sample_input):04d}"
    else:
        sample_id = sample_input

    schema_path = os.path.join(dataset_path, "dataset_schema.json")
    if not os.path.exists(schema_path):
        print(f"Error: Dataset schema not found at {schema_path}")
        return

    with open(schema_path, "r") as f:
        records = json.load(f)

    # Find the target record
    record = next((r for r in records if r["id"] == sample_id), None)
    if not record:
        print(f"Error: Sample {sample_id} not found in dataset.")
        return

    print(f"\n{'='*50}")
    print(f" REPRODUCIBILITY VERIFICATION: {sample_id}")
    print(f" Source Path: {dataset_path}")
    print(f"{'='*50}")
    
    # 1. Re-render the image from saved schema
    temp_img_path = f"repro_{sample_id}.png"
    temp_radar_path = f"repro_{sample_id}_radar.png"
    
    render_schema = {
        "nodes": record["schema"]["nodes"],
        "connections": record["schema"]["connections"]
    }
    render_microstructure(render_schema, size=(128, 128), out_path=temp_img_path)
    print(f"[1/3] Re-rendered image saved to: {temp_img_path}")

    # 2. Re-calculate properties
    print(f"[2/3] Computing properties for verification...")
    new_props = process_image(temp_img_path, silent=True)
    old_props = record["properties"]

    # Generate the new dual-subplot radar chart
    save_radar_chart(
        np.array(new_props['C_eff']), 
        np.array(new_props['kappa_eff']), 
        f"Reproduction: {sample_id}", 
        temp_radar_path
    )
    print(f"      Dual-radar chart saved to: {temp_radar_path}")

    # 3. Compare core properties
    print(f"[3/3] Property Comparison:")
    print("-" * 75)
    print(f"{'Property':<20} | {'Original':<15} | {'Re-calculated':<15} | {'Diff (%)'}")
    print("-" * 75)
    
    metrics = ["volume_fraction", "C11", "C22", "k11"]
    for m in metrics:
        v_old = old_props.get(m, 0)
        v_new = new_props.get(m, 0)
        # Avoid division by zero
        denom = abs(v_old) if abs(v_old) > 1e-9 else 1e-9
        diff = abs(v_old - v_new) / denom * 100
        print(f"{m:<20} | {v_old:<15.6f} | {v_new:<15.6f} | {diff:.4f}%")

    print("-" * 75)
    print(f"Verification Complete. Visuals ready in current directory.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python verify_reproducibility.py <sample_id_or_number> [dataset_path]")
    else:
        target = sys.argv[1]
        path = sys.argv[2] if len(sys.argv) > 2 else "Output/dataset/batch_1"
        verify_sample(target, path)
