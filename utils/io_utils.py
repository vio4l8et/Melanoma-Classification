import cv2
import base64

def image_to_base64(image_rgb, max_width=600):
    # 너무 큰 이미지면 줄이기
    h, w = image_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")
