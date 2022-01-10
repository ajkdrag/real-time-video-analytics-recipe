import json
import importlib
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from argparse import ArgumentParser
from libs.augmentor import augment_multiple
from utils.loader import load_multiple, get_batches
from utils.writer import write_multiple


def parse_config(cfg_path):
    with open(cfg_path) as stream:
        return json.load(stream)


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--bbs_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def proc(
    batch_id,
    batch,
    augmentor,
    imgs_path,
    aug_imgs_path,
    aug_bbs_path,
    prefix,
    rounds,
):
    imgs, bbs = load_multiple(batch, imgs_path)
    if imgs and bbs and len(imgs) == len(bbs):
        for idx in range(rounds):
            aug_imgs, aug_bbs = augment_multiple(imgs, bbs, augmentor)
            write_multiple(
                aug_imgs,
                aug_bbs,
                aug_imgs_path,
                aug_bbs_path,
                f"{prefix}_{idx}_",
            )
            print(f"Finished augmenting batch: {batch_id}, round: {idx}")


def main(FLAGS):
    cfg_file = FLAGS.config
    cfg = parse_config(cfg_file)

    imgs_path = Path(FLAGS.imgs_dir)
    bbs_path = Path(FLAGS.bbs_dir)

    aug_imgs_path = Path(FLAGS.out_dir, "images")
    aug_bbs_path = Path(FLAGS.out_dir, "labels")

    aug_imgs_path.mkdir(parents=True, exist_ok=True)
    aug_bbs_path.mkdir(parents=True, exist_ok=True)

    aug = getattr(importlib.import_module(cfg["augmentor"]), "aug")
    prefix = cfg.get("prefix", datetime.now().strftime("%Y%m%d%H%M%S"))
    rounds = cfg["rounds"]

    params = (aug, imgs_path, aug_imgs_path, aug_bbs_path, prefix, rounds)

    with Pool(cfg["pool_sz"]) as pool:
        pool.starmap(
            proc,
            iterable=(
                (idx, list(batch), *params)
                for idx, batch in enumerate(
                    get_batches(bbs_path.glob("*.txt"), cfg["batch_sz"])
                )
            ),
        )


if __name__ == "__main__":
    flags = parse_flags()
    main(flags)
