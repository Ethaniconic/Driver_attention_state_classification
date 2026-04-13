# Driver Attention State Classifier

Binary image classification project to detect whether a driver appears drowsy or not drowsy.

## Overview

This repository contains:

- Dataset split utility for preparing train and test folders
- Model training workflow in a Jupyter notebook
- Real-time webcam inference script using a fine-tuned EfficientNet-B0 model

The classifier predicts one of two classes:

- Drowsy
- Non Drowsy

## Project Structure

```text
Driver attention state classifier/
├── camera.py
├── data.py
├── model.ipynb
├── requirements.txt
├── data/
│   ├── Driver_drowsiness_dataset/
│   │   ├── Drowsy/
│   │   └── Non Drowsy/
│   ├── train/
│   └── test/
└── models/
    └── drowsy_or_not.pth
```

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- OpenCV
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (recommended): use a virtual environment first.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

Place raw images in:

- `data/Driver_drowsiness_dataset/Drowsy`
- `data/Driver_drowsiness_dataset/Non Drowsy`

Create train/test splits:

```bash
python data.py
```

By default, the script uses an 80/20 train-test split.

## Run Real-Time Inference

Make sure the model weights exist at `models/drowsy_or_not.pth`, then run:

```bash
python camera.py
```

Controls:

- Press `q` to quit the webcam window.

## Notes

- `camera.py` uses `data/train` class folder names to map prediction indices to labels.
- If camera index `0` is unavailable, the script attempts index `1` automatically.

## Future Improvements

- Add evaluation metrics and confusion matrix reporting
- Add command-line arguments for camera index and model path
- Add model training and export scripts outside notebook format
- Add CI checks for linting and formatting

## License

Choose and add a license file (for example, MIT) before publishing publicly.
