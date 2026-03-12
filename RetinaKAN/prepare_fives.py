import os
import shutil

root = "datasets/FIVES"
splits = ["train", "test"]

valid_ext = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

for split in splits:
    img_dir = os.path.join(root, split, "Original")
    mask_dir = os.path.join(root, split, "Ground truth")

    for filename in os.listdir(img_dir):

        # Skip hidden/system files
        if not any(filename.lower().endswith(ext) for ext in valid_ext):
            continue

        new_name = split + "_" + filename

        src_img = os.path.join(img_dir, filename)
        src_mask = os.path.join(mask_dir, filename)

        if not os.path.exists(src_mask):
            print(f"Mask missing for {filename}")
            continue

        shutil.copy(src_img, os.path.join(root, "images", new_name))
        shutil.copy(src_mask, os.path.join(root, "masks", new_name))

print("Done merging FIVES safely.")
