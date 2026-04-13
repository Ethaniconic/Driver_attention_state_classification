from pathlib import Path
import random
import shutil

def split_dataset(ratio_train: float = 0.8, seed: int = 42, clear_existing: bool = True) -> None:
    data_root = Path("data")
    src_root = data_root / "Driver_drowsiness_dataset"
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    random.seed(seed)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for cls_name in ["Drowsy", "Non Drowsy"]:
        cls_path = src_root / cls_name
        train_cls_dir = train_dir / cls_name
        test_cls_dir = test_dir / cls_name
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        test_cls_dir.mkdir(parents=True, exist_ok=True)

        if clear_existing:
            for old_img in train_cls_dir.iterdir():
                if old_img.is_file():
                    old_img.unlink()
            for old_img in test_cls_dir.iterdir():
                if old_img.is_file():
                    old_img.unlink()

        images = [
            p
            for p in cls_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if not images:
            continue

        random.shuffle(images)

        n_train = int(len(images) * ratio_train)
        train_imgs, test_imgs = images[:n_train], images[n_train:]

        for img in train_imgs:
            shutil.copy(img, train_cls_dir / img.name)

        for img in test_imgs:
            shutil.copy(img, test_cls_dir / img.name)

    print(f"train/test split done | seed={seed} | ratio_train={ratio_train}")


if __name__ == "__main__":
    split_dataset()