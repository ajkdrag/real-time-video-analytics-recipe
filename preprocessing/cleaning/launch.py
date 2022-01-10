from pathlib import Path
from distutils.dir_util import copy_tree
from argparse import ArgumentParser


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--bbs_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def add_empty_labels(all_img_files, bbs_path):
    for file in all_img_files:
        corr_bbs_file = bbs_path / f"{file.stem}.txt"
        if not corr_bbs_file.exists():
            corr_bbs_file.touch()


def rename_file(file_):
    new_name = str(file_).replace(" ", "").replace("(", "_").replace(")", "")
    file_.rename(new_name)


def rename_all_files(all_img_files, all_bbs_files):
    for img, bbs in zip(all_img_files, all_bbs_files):
        rename_file(img)
        rename_file(bbs)


def main(FLAGS):
    imgs_path = Path(FLAGS.imgs_dir)
    bbs_path = Path(FLAGS.bbs_dir)

    cleaned_imgs_path = Path(FLAGS.out_dir, "images")
    cleaned_bbs_path = Path(FLAGS.out_dir, "labels")

    cleaned_imgs_path.mkdir(parents=True, exist_ok=True)
    cleaned_bbs_path.mkdir(parents=True, exist_ok=True)

    copy_tree(str(imgs_path), str(cleaned_imgs_path))
    copy_tree(str(bbs_path), str(cleaned_bbs_path))

    all_img_files = list(cleaned_imgs_path.iterdir())
    add_empty_labels(all_img_files, cleaned_bbs_path)

    all_bbs_files = list(cleaned_bbs_path.iterdir())
    assert len(all_img_files) == len(all_bbs_files)

    rename_all_files(all_img_files, all_bbs_files)


if __name__ == "__main__":
    flags = parse_flags()
    main(flags)
