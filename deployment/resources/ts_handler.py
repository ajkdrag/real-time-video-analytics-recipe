import time
import torch
import torchvision
import numpy as np
from cv2 import cv2
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    Custom model handler implementation.
    """

    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.batch_size = 1
        self.inference_img_sz = [640, 640]
        self.explain_img_sz = [160, 160]

    def preprocess(self, request_data):
        try:
            nparr = np.asarray(bytearray(request_data[0]["body"]), dtype=np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.og_shape = decoded.shape[:2]
            img_sz = (
                self.explain_img_sz if self._is_explain() else self.inference_img_sz
            )
            resized, _, _ = letterbox(decoded, img_sz, auto=False)
            img = resized[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img).unsqueeze(0)
        except Exception as exc:
            print(exc)

    def postprocess(self, inference_output, explain=False):
        postprocess_output = inference_output[0]
        img_sz = self.explain_img_sz if explain else self.inference_img_sz
        pred = non_max_suppression(
            postprocess_output, self.og_shape, img_sz, conf_thres=0.2
        )
        pred = [p.tolist() for p in pred]
        return [pred]


def letterbox(
    im,
    new_shape,
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def non_max_suppression(
    prediction, og_shape, resized_shape, conf_thres=0.5, iou_thres=0.6, agnostic=False
):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Number of classes.
    nc = prediction[0].shape[1] - 5

    # Candidates.
    xc = prediction[..., 4] > conf_thres

    # Settings:
    # Minimum and maximum box width and height in pixels.
    min_wh, max_wh = 2, 256

    # Maximum number of detections per image.
    max_det = 100

    # Timeout.
    time_limit = 10.0

    # Multiple labels per box (adds 0.5ms/img).
    multi_label = nc > 1

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply constraints:
        # Confidence.
        x = x[xc[xi]]

        # If none remain process next image.
        if not x.shape[0]:
            continue

        # Compute conf.
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2).
        box = xywh2xyxy(*og_shape, *resized_shape, x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls).
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            # Best class only.
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # If none remain process next image.
        # Number of boxes.
        n = x.shape[0]
        if not n:
            continue

        # Batched NMS:
        # Classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS.
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        # Limit detections.
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(origin_h, origin_w, rsz_h, rsz_w, x):
    y = torch.zeros_like(x)
    r_w = rsz_w / origin_w
    r_h = rsz_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (rsz_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (rsz_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (rsz_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (rsz_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h
    return y
