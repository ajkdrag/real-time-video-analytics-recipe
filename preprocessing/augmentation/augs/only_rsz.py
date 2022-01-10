from imgaug import augmenters as iaa


aug = iaa.Resize({"height": 360, "width": 640})