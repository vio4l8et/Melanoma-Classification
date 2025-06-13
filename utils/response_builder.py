def build_summary(cls, prob, risk, features):
    return (
        f"분류 결과: {cls} ({prob:.2f}) → 위험도: {risk}\n"
        f"비대칭성: {features['asymmetry']:.4f}\n"
        f"경계 불규칙성: {features['border_irregularity']:.4f}\n"
        f"색상 다양성: {features['color_variation']:.4f}\n"
        f"크기: {features['diameter_mm']:.2f} mm"
    )
