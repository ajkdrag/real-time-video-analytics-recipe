import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(42)

sometimes = lambda aug: iaa.Sometimes(0.3, aug)


aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        sometimes(
            iaa.KeepSizeByResize(
                iaa.OneOf(
                    [
                        iaa.Crop(
                            percent=(
                                [0.05, 0.1],
                                [0.05, 0.1],
                                [0.05, 0.1],
                                [0.05, 0.1],
                            ),
                            keep_size=False,
                        ),
                        iaa.CropAndPad(
                            percent=(
                                [0.05, 0.1],
                                [0.05, 0.1],
                                [0.05, 0.1],
                                [0.05, 0.1],
                            ),
                            keep_size=False,
                        ),
                    ]
                ),
                interpolation="nearest",
            )
        ),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2),
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                },  # translate by -20 to +20 percent (per axis)
                order=[0, 1,],  # use nearest neighbour or bilinear interpolation (fast)
                cval=0,  # if mode is constant, use a cval between 0 and 255
            )
        ),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                iaa.OneOf(
                    [
                        iaa.imgcorruptlike.MotionBlur(severity=2),
                        iaa.imgcorruptlike.DefocusBlur(severity=2),
                        iaa.imgcorruptlike.Brightness(severity=2),
                    ]
                ),
                iaa.OneOf(
                    [
                        iaa.pillike.Equalize(),
                        iaa.pillike.Autocontrast(),
                        iaa.pillike.EnhanceSharpness(),
                    ]
                ),
            ],
            random_order=True,
        ),
    ],
    random_order=True,
)
