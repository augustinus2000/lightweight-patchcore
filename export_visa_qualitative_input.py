from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np

# ---------------------------------------------------------
# 여기에 원하는 이미지와 GT mask 경로 입력
# ---------------------------------------------------------
IMAGE_PATH = "/home/jhim/Patchcore/patchcore-inspection/data/visa_pytorch/1cls/fryum/test/bad/001.JPG"
MASK_PATH  = "/home/jhim/Patchcore/patchcore-inspection/data/visa_pytorch/1cls/fryum/ground_truth/bad/001.png"

OUTPUT_DIR = Path("qualitative_export")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 실제 VisaDataset과 동일한 geometric transform
img_geom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

mask_geom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# ---------------------------------------------------------
# Input image
# ---------------------------------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_processed = img_geom_transform(image)

image_processed.save(
    OUTPUT_DIR / "input_224.png"
)

# ---------------------------------------------------------
# Ground-truth mask
# ---------------------------------------------------------
mask = Image.open(MASK_PATH)
mask_processed = mask_geom_transform(mask)

mask_processed.save(
    OUTPUT_DIR / "gt_224.png"
)

print("Original image size :", image.size)
print("Processed image size:", image_processed.size)
print("Original mask size  :", mask.size)
print("Processed mask size :", mask_processed.size)

print()
print("Saved:")
print(OUTPUT_DIR / "input_224.png")
print(OUTPUT_DIR / "gt_224.png")
