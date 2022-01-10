from pathlib import Path
from cv2 import cv2
from utils.general import xywh2xyxy
from itertools import islice, chain


def load_bbs(bbs: Path, img_h, img_w):
    bboxes = []
    try:
        for line in bbs.open():
            yolo_bbox = [token.strip() for token in line.split(" ") if token.strip()]
            class_id = yolo_bbox[0]
            xyxy = xywh2xyxy(yolo_bbox[1:], img_h, img_w)
            bboxes.append([class_id, *xyxy])
        return bboxes
    except Exception as exc:
        print(f"Failed to load annotation: {bbs}, Reason: {exc}")


def load_image(image: Path, alt="png"):
    img = cv2.imread(str(image), 1)
    if img is None:
        alt_image = image.parent / f"{image.stem}.{alt}"
        img = cv2.imread(str(alt_image), cv2.COLOR_BGR2RGB)
        if img is None:
            print(f"Failed to load image (jpg/png): {image}")
    return img


def load_multiple(bbs_paths: list, img_dir: Path, ext="jpg"):
    images = []
    bbss = []

    for bbs_path in bbs_paths:
        img_path = img_dir / f"{bbs_path.stem}.{ext}"
        loaded_img = load_image(img_path)
        if loaded_img is not None:
            img_h, img_w = loaded_img.shape[:2]
            loaded_bbs = load_bbs(bbs_path, img_h, img_w)
            if loaded_img is not None and load_bbs is not None:
                images.append((img_path.name, loaded_img))
                bbss.append((bbs_path.name, loaded_bbs))

    return images, bbss


def get_batches(iterable, bs):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, bs - 1))
