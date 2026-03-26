import json
import os

json_path = "Output/dataset/batch_1/dataset_schema.json"
if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

print(f"Loading {json_path}...")
with open(json_path, "r") as f:
    data = json.load(f)

count = 0
for record in data:
    if "radar_path" in record:
        # Replace 'radars' folder with 'images' folder in the path string
        old_path = record["radar_path"]
        new_path = old_path.replace("\\radars\\", "\\images\\").replace("/radars/", "/images/")
        if old_path != new_path:
            record["radar_path"] = new_path
            count += 1

print(f"Updated {count} records.")
print("Saving updated JSON...")
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("Done.")
