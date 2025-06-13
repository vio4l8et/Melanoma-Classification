from flask import request
from PIL import Image
import numpy as np
import cv2

def receive_and_preprocess_image(max_width=800):
    try:
        # 1. 이미지 파일 수신
        file = request.files['image']

        # 2. PIL 이미지 열기 (RGB로 변환)
        image_pil = Image.open(file.stream).convert('RGB')
        image_np = np.array(image_pil)

        # 3. 이미지 크기 축소 (너비 기준)
        h, w = image_np.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            image_np = cv2.resize(image_np, (new_w, new_h))

            # PIL 이미지도 맞춰서 다시 만들기 (선택)
            image_pil = Image.fromarray(image_np)

        return image_pil, image_np

    except Exception as e:
        raise ValueError(f"이미지 수신 또는 전처리 오류: {str(e)}")

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    return image_pil, image_np