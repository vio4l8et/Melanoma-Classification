import numpy as np
import cv2
from features.feature_utils import (
    asymmetry_from_mask,
    border_irregularity_from_mask,
    color_variation_from_rgb,
    diameter_from_mask
)

def get_lesion_mask(image_np, box):
    x, y, w, h = box
    lesion_crop = image_np[y:y+h, x:x+w]
    gray = cv2.cvtColor(lesion_crop, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    full_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
    full_mask[y:y+h, x:x+w] = binary.astype(np.float32)
    return full_mask

def extract_features(image_np, mask):
    return {
        "asymmetry": asymmetry_from_mask(mask),
        "border_irregularity": border_irregularity_from_mask(mask),
        "color_variation": color_variation_from_rgb(image_np, mask),
        "diameter_mm": diameter_from_mask(mask)
    }
