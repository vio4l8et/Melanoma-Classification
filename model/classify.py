import torch
from torchvision import transforms

model_input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7603, 0.5931, 0.5682],
                         std=[0.1943, 0.1956, 0.2142])
])

def predict_class(model, image_pil, device):
    x = model_input_transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).item()
        pred = 1 if prob >= 0.5 else 0
        class_name = "melanoma" if pred == 1 else "non-melanoma"
        conf = prob if pred == 1 else 1 - prob

        if conf >= 0.9:
            risk = "위험"
        elif conf >= 0.7:
            risk = "주의"
        else:
            risk = "안전"

    return class_name, conf, risk
