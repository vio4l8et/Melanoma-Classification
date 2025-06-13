from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import torch

# 사용자 정의 모듈 불러오기
from model.convnext_model import load_model
from model.classify import predict_class
from image_processing.receive_image import receive_and_preprocess_image, load_image
from image_processing.lesion_detect import detect_lesion
from image_processing.draw_bbox import draw_box
from features.extract_features import get_lesion_mask, extract_features
from utils.io_utils import image_to_base64
from utils.response_builder import build_summary
from PIL import Image
import numpy as np
import os

# Flask 앱 초기화 및 비밀키 설정
app = Flask(__name__)
app.secret_key = "your-secret-key"  # 실제 배포 시 보안 강화 필요

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("best_model.pth", device=device)

# --- 페이지 라우팅 ---
@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        username = request.form['username']
        session['username'] = username
        return redirect(url_for('main'))
    return render_template('main.html')

@app.route('/sample-select')
def sample_select():
    username = session.get('username', '사용자')
    return render_template('sample_select.html', username=username)

# --- 분석 요청 처리 ---
@app.route("/sample-analyze", methods=["POST"])
def sample_analyze():
    try:
        image_path = request.form.get("image_path")
        if not image_path:
            return "이미지 경로가 제공되지 않았습니다.", 400

        # 상대 경로로부터 절대 경로 구성
        static_root = os.path.join(app.root_path, 'static')
        full_image_path = os.path.join(static_root, image_path)

        # 이미지 로딩 및 분석
        image_pil, image_np = load_image(full_image_path)
        cls, prob, risk = predict_class(model, image_pil, device)
        box, _ = detect_lesion(image_np)
        boxed_img = draw_box(image_np, box)
        lesion_mask = get_lesion_mask(image_np, box)
        features = extract_features(image_np, lesion_mask)
        boxed_base64 = image_to_base64(boxed_img)

        result = {
            "username": session.get("username", "사용자"),
            "final_class": cls,
            "risk": risk,
            "trust": round(prob * 100, 1),
            "boxed_image": boxed_base64,
            "features": {
                "asymmetry": float(features["asymmetry"]),
                "color": float(features["color_variation"]),
                "border": float(features["border_irregularity"]),
                "diameter": float(features["diameter_mm"]),
            }
        }

        # 세션/스토리지 저장이 불가하므로 JS에서 저장하도록 처리
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_image():
    try:
        # username 세션 저장
        session['username'] = request.form.get('username', '사용자')

        # 1. 이미지 수신
        image_pil, image_np = receive_and_preprocess_image()
        image_pil.save("temp_uploaded.jpg")

        # 2. 분류 예측
        cls, prob, risk = predict_class(model, image_pil, device)

        # 3. 병변 감지 및 시각화
        box, _ = detect_lesion(image_np)
        boxed_img = draw_box(image_np, box)

        # 4. 특징 추출
        lesion_mask = get_lesion_mask(image_np, box)
        features = extract_features(image_np, lesion_mask)

        # 5. 시각화 및 요약
        summary = build_summary(cls, prob, risk, features)
        boxed_base64 = image_to_base64(boxed_img)

        # 6. 결과 JSON 반환 (세션 저장 X)
        return jsonify({
            "summary": summary,
            "class": cls,
            "risk": risk,
            "prob": float(prob),  # ← 여기 중요
            "boxed_image": boxed_base64,
            "features": {
                "asymmetry": float(features["asymmetry"]),
                "color": float(features["color_variation"]),
                "border": float(features["border_irregularity"]),
                "diameter": float(features["diameter_mm"])
            },
            "username": session.get("username", "사용자"),
            "final_class": cls,
            "trust": round(float(prob) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- 결과 페이지 라우팅 ---
@app.route("/melanoma_result")
def melanoma_result():
    return render_template("melanoma_result.html")

@app.route("/benign_result")
def benign_result():
    return render_template("benign_result.html")

# --- 실행 ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
