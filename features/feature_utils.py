import cv2
import numpy as np

def asymmetry_from_mask(mask):
    H, W = mask.shape
    mid_W = W // 2
    mid_H = H // 2
    left = mask[:, :mid_W]
    right = np.fliplr(mask[:, -mid_W:])
    top = mask[:mid_H, :]
    bottom = np.flipud(mask[-mid_H:, :])
    return np.mean((left - right) ** 2) + np.mean((top - bottom) ** 2)

def border_irregularity_from_mask(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    return (perimeter ** 2) / (4 * np.pi * area + 1e-6)

def color_variation_from_rgb(image_rgb, mask):
    mask = mask.astype(bool)
    pixels = image_rgb[mask]
    return np.std(pixels) if len(pixels) > 0 else 0.0

def diameter_from_mask(mask, pixel_spacing_mm=0.2):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    return radius * 2 * pixel_spacing_mm
