import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path

# Load the Attention Classifier Model (EfficientNet-B0 fine‑tuned for 2 classes)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model_path = Path('models/drowsy_or_not.pth')
if not model_path.exists():
    raise FileNotFoundError(f'Classifier model not found at {model_path}')
state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Class names – try to read from training folder, fallback to defaults
try:
    class_names = datasets.ImageFolder('data/train').classes
except Exception:
    class_names = ['Drowsy', 'Non Drowsy']

# Pre‑processing pipeline for the cropped face (same as training)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize OpenCV YuNet face detector (ONNX model)
# The model is downloaded by download_model.py into models/face_detection_yunet_2023mar.onnx
yunet_path = Path('models/face_detection_yunet_2023mar.onnx')
if not yunet_path.exists():
    raise FileNotFoundError(f'YuNet model not found at {yunet_path}. Run download_model.py first.')

# We'll create the detector after we know the frame size

def init_yunet(frame_width, frame_height):
    # score_threshold, nms_threshold, top_k are typical defaults for YuNet
    return cv2.FaceDetectorYN_create(str(yunet_path), '', (frame_width, frame_height), 0.9, 0.3, 5000)

def crop_face(frame, bbox, padding=0.25):
    """Crop the face region with optional padding.
    bbox: (x, y, w, h) as returned by YuNet.
    """
    h, w, _ = frame.shape
    x, y, bw, bh = bbox
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    x1 = max(0, int(x) - pad_w)
    y1 = max(0, int(y) - pad_h)
    x2 = min(w, int(x + bw) + pad_w)
    y2 = min(h, int(y + bh) + pad_h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError('Could not open camera. Check webcam permission/device index.')

# Grab an initial frame to initialise the detector with the correct size
ret, init_frame = cap.read()
if not ret:
    raise RuntimeError('Failed to read initial frame from camera.')
frame_h, frame_w = init_frame.shape[:2]
face_detector = init_yunet(frame_w, frame_h)

update_interval = 30  # Update label every 60 frames (original behavior)
frame_counter = 0
stable_idx = 0
stable_conf = 0.0

print('Starting webcam – press "q" to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Mirror view for user friendliness
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Detect faces – returns (retval, faces) where faces is Nx15 array
    retval, faces = face_detector.detect(frame)
    if retval and faces is not None and len(faces) > 0:
        # Use the first (most confident) detection
        x, y, w, h = faces[0][:4]
        face_crop, coords = crop_face(frame, (x, y, w, h))
        if face_crop.size > 0:
            # Preprocess and run classifier
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = preprocess(face_rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred = model(tensor)
                prob = torch.softmax(pred, dim=1)[0]
                idx = prob.argmax().item()
                conf = prob[idx].item()
            frame_counter += 1
            if frame_counter % update_interval == 0:
                stable_idx = idx
                stable_conf = conf
            # Draw bounding box and label
            x1, y1, x2, y2 = coords
            color = (0, 255, 0) if class_names[stable_idx] == 'Non Drowsy' else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[stable_idx]} ({stable_conf:.2f})"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(display, 'No Face Detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Driver Attention ADAS', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# FaceDetectorYN does not require explicit close; resources are released with cap.release()