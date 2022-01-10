import ast
import requests
import numpy as np
from cv2 import cv2
from argparse import ArgumentParser
from collections import deque

COLORS = {0: "blue", 1: "green", 2: "red"}
LIGHT_CLASS = 0
BAT_CLASS = 1


def detect_color(frame, bbox):
    x1, x2, y1, y2 = bbox[:4]
    roi = frame[x1:x2, y1:y2]
    bgr_means = [np.mean(roi[:, :, 0]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 2])]
    index_max = int(max(range(3), key=bgr_means.__getitem__))
    return COLORS[index_max]


def prepare(frame):
    _, encoded_image = cv2.imencode(".jpg", frame)
    return encoded_image.tobytes()


def fire(url, content):
    files = {"body": content}
    return ast.literal_eval(requests.post(url, files=files).text)[0]


def clean_bat_bboxes(bbox_dict):
    """
        bboxes: [x1, y1, x2, y2, score, area] 
    """
    bbox_dict.setdefault(BAT_CLASS, [])
    bboxes = bbox_dict.pop(BAT_CLASS)
    count = len(bboxes)
    bat_bbox = []
    if count > 0:
        best = max(range(count), key=lambda x: bboxes[x][5])
        bat_bbox.append(bboxes[best])
    bbox_dict["bat"] = bat_bbox


def clean_light_bboxes(bbox_dict, frame):
    bbox_dict.setdefault(LIGHT_CLASS, [])
    light_bboxes = bbox_dict.pop(LIGHT_CLASS)
    bat_bbox = bbox_dict["bat"]

    def is_light_inside_bat(light_bbox, bat_bbox):
        return (
            bat_bbox[0] <= light_bbox[0] <= bat_bbox[2]
            and bat_bbox[1] <= light_bbox[1] <= bat_bbox[3]
        )

    if len(bat_bbox) == 1:
        light_bboxes = [
            bbox for bbox in light_bboxes if is_light_inside_bat(bbox, bat_bbox[0])
        ]

    for bbox in light_bboxes:
        color = detect_color(frame, bbox)
        bbox_dict.setdefault(color, []).append(bbox)


def annotate(bbox_dict, frame):
    for class_, bboxes in bbox_dict.items():
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            display_str = "{}: {}%".format(class_, round(100 * bbox[4]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220))
            cv2.putText(
                frame,
                display_str,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def process(frame, result):
    bbox_dict = {}
    for bbox in result:
        x1, y1, x2, y2 = list(map(int, bbox[:-2]))
        area = (x2 - x1) * (y2 - y1)
        score, class_ = bbox[-2:]
        bbox_dict.setdefault(class_, [])
        bbox_dict[class_].append([x1, y1, x2, y2, score, area])

    clean_bat_bboxes(bbox_dict)
    clean_light_bboxes(bbox_dict, frame)
    annotate(bbox_dict, frame)


def main(FLAGS):
    url = f"http://localhost:8080/predictions/{FLAGS.model}"
    _ = deque(maxlen=FLAGS.seq_size)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        content = prepare(frame)
        resp = fire(url, content)
        process(frame, resp)

        cv2.imshow(str("Inference"), frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break

    cap.release()


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5_exp_1")
    parser.add_argument("--seq_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    flags = parse_flags()
    main(flags)