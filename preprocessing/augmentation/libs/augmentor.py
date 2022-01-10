import numpy as np

from imgaug.augmentables.batches import Batch
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def list_bbs_to_imgaug_bbs(list_bbs, img_h, img_w):
    bbs = BoundingBoxesOnImage(
        [
            BoundingBox(x1, y1, x2, y2, str(class_id))
            for class_id, x1, y1, x2, y2 in list_bbs
        ],
        shape=(img_h, img_w),
    )
    return bbs


def clean_all_bbs(list_imgaug_bbs):
    return [
        imgaug_bbs.remove_out_of_image_fraction(0.7).clip_out_of_image()
        for imgaug_bbs in list_imgaug_bbs
    ]


def augment_multiple(batch_imgs, batch_bbs, aug):
    imgs_zipped = list(zip(*batch_imgs))
    bbs_zipped = list(zip(*batch_bbs))

    batch_img_names = imgs_zipped[0]
    batch_bbs_names = bbs_zipped[0]

    batch_imgs_only = imgs_zipped[1]
    batch_bbs_only = [
        list_bbs_to_imgaug_bbs(el, *batch_imgs_only[idx].shape[:2])
        for idx, el in enumerate(bbs_zipped[1])
    ]

    batch = Batch(images=np.array(batch_imgs_only), bounding_boxes=batch_bbs_only)
    augmented = aug.augment_batch_(batch)

    return (
        zip(batch_img_names, augmented.images_aug),
        zip(batch_bbs_names, clean_all_bbs(augmented.bounding_boxes_aug)),
    )

