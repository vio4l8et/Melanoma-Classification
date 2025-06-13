import cv2
import numpy as np

def detect_lesion(image_rgb):
    h, w, _ = image_rgb.shape
    center = (w // 2, h // 2)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("병변이 감지되지 않았습니다.")

    def contour_distance_to_center(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.linalg.norm(np.array([cx, cy]) - np.array(center))

    central_contour = min(contours, key=contour_distance_to_center)
    x, y, w, h = cv2.boundingRect(central_contour)
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w = min(w + 2 * pad_x, image_rgb.shape[1] - x)
    h = min(h + 2 * pad_y, image_rgb.shape[0] - y)

    return (x, y, w, h), central_contour
