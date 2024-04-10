import numpy as np


def hard_NMS(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def run_NMS(prediction, iou_threshold, maxDets=100):
    keep = np.zeros(len(prediction), dtype=np.int)
    for cls_id in np.unique(prediction[:, 0]):
        inds = np.where(prediction[:, 0] == cls_id)[0]
        if len(inds) == 0:
            continue
        cls_boxes = prediction[inds, 1:5]
        cls_scores = prediction[inds, 5]
        cls_keep = hard_NMS(
            boxes=cls_boxes, scores=cls_scores, iou_threshold=iou_threshold
        )
        keep[inds[cls_keep]] = 1
    prediction = prediction[np.where(keep > 0)]
    order = prediction[:, 5].argsort()[::-1]
    return prediction[order[:maxDets]]
