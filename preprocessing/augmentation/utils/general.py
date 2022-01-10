def xywh2xyxy(xywh, img_h, img_w):
    x, y, w, h = [float(val) for val in xywh]
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h

    tl_and_br = [x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5]
    return list(map(int, tl_and_br))


def xyxy2xywh(xyxy, img_h, img_w):
    xmin, ymin, xmax, ymax = [float(val) for val in xyxy]
    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x = (xmin + xmax) * 0.5 - 1
    y = (ymin + ymax) * 0.5 - 1
    w = xmax - xmin
    h = ymax - ymin
    return [x * dw, y * dh, w * dw, h * dh]

