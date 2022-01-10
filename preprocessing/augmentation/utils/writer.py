from pathlib import Path
from cv2 import cv2
from utils.general import xyxy2xywh


def write_bbs(bbs, img_h, img_w, out_path: Path):
    with out_path.open(mode="w") as stream:
        for bb in bbs:
            xywh = xyxy2xywh([bb.x1, bb.y1, bb.x2, bb.y2], img_h, img_w)
            stream.write(" ".join(list(map(str, [bb.label, *xywh]))) + "\n")


def write_image(image, out_path: Path):
    cv2.imwrite(str(out_path), image)


def write_multiple(
    images_aug, bbss_aug, images_dir: Path, bbss_dir: Path, prefix: str
):
    for img_tup, bbs_tup in zip(images_aug, bbss_aug):
        img_out_path = images_dir / (prefix + img_tup[0])
        bbs_out_path = bbss_dir / (prefix + bbs_tup[0])
        img_h, img_w = img_tup[1].shape[:2]
        write_image(img_tup[1], img_out_path)
        write_bbs(bbs_tup[1], img_h, img_w, bbs_out_path)
