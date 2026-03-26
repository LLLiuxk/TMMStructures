import json
import numpy as np
from plot_combined_radar import save_radar_chart

def regen():
    print("Regenerating correct radars...")
    with open('output/dataset/test/results.json', 'r') as f:
        data = json.load(f)
    for item in data:
        C_eff = np.array(item['properties']['C_eff'])
        kappa_eff = np.array(item['properties']['kappa_eff'])
        out_path = item['radar_path']
        save_radar_chart(C_eff, kappa_eff, item['id'], out_path)
        print(f"  Saved {out_path}")

    print("Regenerating buggy radars...")
    with open('output/dataset/test/results_buggy.json', 'r') as f:
        data_buggy = json.load(f)
    for item in data_buggy:
        C_eff = np.array(item['C_eff'])
        kappa_eff = np.array(item['kappa_eff'])
        out_path = item['radar_path']
        save_radar_chart(C_eff, kappa_eff, f"{item['id']} (Buggy)", out_path)
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    regen()
