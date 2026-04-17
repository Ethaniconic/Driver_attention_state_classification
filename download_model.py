import urllib.request
from pathlib import Path

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File already exists at {dest_path}")
        return
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")

def main():
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # MediaPipe face detector (fallback, not used in final implementation but kept for reference)
    mp_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    mp_path = models_dir / "blaze_face_short_range.tflite"
    download_file(mp_url, mp_path)

    # OpenCV YuNet face detector model (ONNX)
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
    download_file(yunet_url, yunet_path)

if __name__ == "__main__":
    main()