import json

CLASSES = ["glass", "metal", "paper", "plastic", "general"]

idx_to_class = {i: name for i, name in enumerate(CLASSES)}

with open("class_map.json", "w") as f:
    json.dump(idx_to_class, f)

print("Wrote class_map.json:", idx_to_class)
