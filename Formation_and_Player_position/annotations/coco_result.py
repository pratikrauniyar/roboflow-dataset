import json
import os
import shutil

def coco_to_per_image_json(coco_json_path, output_dir="output_json"):
    # Remove previous results if output_dir exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO annotation file
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build category lookup (id -> name)
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Prepare offense and defense id sets
    defense_ids = set(range(1, 14))
    offense_ids = set(cat_id for cat_id in categories if cat_id >= 14)

    # Group annotations per image
    anns_per_image = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        cls_name = categories.get(category_id, "unknown")

        # Extract keypoints if present
        keypoints = ann.get("keypoints", [])
        xy_pairs = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 3)] if keypoints else []

        if image_id not in anns_per_image:
            anns_per_image[image_id] = {
                "cls": set(),
                "cls_off": set(),
                "cls_def": set(),
                "x_def": [],
                "y_def": [],
                "x_off": [],
                "y_off": []
            }

        anns_per_image[image_id]["cls"].add(cls_name)
        if category_id in defense_ids:
            anns_per_image[image_id]["cls_def"].add(cls_name)
            for x, y in xy_pairs:
                anns_per_image[image_id]["x_def"].append(x)
                anns_per_image[image_id]["y_def"].append(y)
        elif category_id in offense_ids:
            anns_per_image[image_id]["cls_off"].add(cls_name)
            for x, y in xy_pairs:
                anns_per_image[image_id]["x_off"].append(x)
                anns_per_image[image_id]["y_off"].append(y)

    # Process each image and create JSON file
    for img in coco_data.get("images", []):
        img_id = img["id"]
        filename = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".json"
        file_path = os.path.join(output_dir, filename)

        # Get annotation data for this image (default empty lists)
        ann_data = anns_per_image.get(img_id, {
            "cls": set(),
            "cls_off": set(),
            "cls_def": set(),
            "x_def": [],
            "y_def": [],
            "x_off": [],
            "y_off": []
        })

        img_json = {
            "image_path": img.get("file_name", ""),
            "x_off": ann_data["x_off"],
            "y_off": ann_data["y_off"],
            "x_def": ann_data["x_def"],
            "y_def": ann_data["y_def"],
            "cls": sorted(list(ann_data["cls"])),
            "cls_off": sorted(list(ann_data["cls_off"])),
            "cls_def": sorted(list(ann_data["cls_def"]))
        }

        with open(file_path, "w") as f:
            json.dump(img_json, f, indent=4)

    print(f"Per-image JSON files saved in '{output_dir}'")

if __name__ == "__main__":
    coco_json_path = "_annotations.coco.json"  # Change to your COCO annotation file path
    output_dir = "annotations_per_image"      # Folder for saving per-image JSON files
    coco_to_per_image_json(coco_json_path, output_dir)