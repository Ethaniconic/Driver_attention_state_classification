import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

model = models.efficientnet_b0(weights=None)
num_classes = 2
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

state_dict = torch.load("models/drowsy_or_not.pth", weights_only=True, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class_names = datasets.ImageFolder("data/train").classes

inference_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def draw_face_guide(frame):
    h, w = frame.shape[:2]

    # Dataset images are square (227x227), so use a centered square guide.
    guide_size = int(min(w, h) * 0.62)
    cx, cy = w // 2, h // 2

    x1 = max(cx - guide_size // 2, 0)
    y1 = max(cy - guide_size // 2, 0)
    x2 = min(cx + guide_size // 2, w - 1)
    y2 = min(cy + guide_size // 2, h - 1)

    # Face outline where users should place their face.
    cv2.ellipse(
        frame,
        (cx, cy),
        (guide_size // 3, int(guide_size * 0.42)),
        0,
        0,
        360,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
    cv2.putText(
        frame,
        "Align face inside outline",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 255),
        2,
    )

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check webcam permission/device index.")

update_interval = 30  # use 60 for slower label updates
frame_count = 0
stable_class_idx = 0
stable_confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flipped_frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    tensor = inference_tf(rgb_frame).unsqueeze(0).to(device)

    # 4. Run predictions
    with torch.inference_mode():
        pred = model(tensor)
        prob = torch.softmax(pred, dim=1)[0]
        class_idx = prob.argmax().item()
        confidence = prob[class_idx].item()

    frame_count += 1
    if frame_count == 1 or frame_count % update_interval == 0:
        stable_class_idx = class_idx
        stable_confidence = confidence

    draw_face_guide(flipped_frame)

    # 5. Annotate frame
    label = f"{class_names[stable_class_idx]} ({stable_confidence:.2f})"
    cv2.putText(flipped_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 6. Last part will be to show the frame
    cv2.imshow('Drowsy or Not', flipped_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()