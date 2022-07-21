import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

min_size = 3
max_candidates = 100

def boxes_from_bitmap(_bitmap, dest_width, dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    '''
    # _bitmap = np.sum(_bitmap.cpu().numpy(), axis=0)
    _bitmap = _bitmap.cpu().numpy()
    # print('_bitmap size(0): ', _bitmap.shape)
    # _bitmap = np.ones(_bitmap.shape) - _bitmap
    _bitmap = np.expand_dims(_bitmap, axis=0)
    # print('_bitmap size(0): ', _bitmap.shape)
    # print('_bitmap: ', _bitmap)
    assert _bitmap.shape[0] == 1
    # assert _bitmap.size(0) == 1
    bitmap = _bitmap[0]  # The first channel
    # bitmap = _bitmap.cpu().numpy()[0]  # The first channel

    height, width = bitmap.shape
    contours, _ = cv2.findContours(
        (bitmap*255).astype(np.uint8),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), max_candidates)
    boxes = np.zeros((num_contours, 4, 2), dtype=np.int32)

    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)

        box = points
        box = points.reshape(-1, 1, 2)
        # box = unclip(points).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        # if sside < min_size + 2:
        #     continue
        box = np.array(box)
        # if not isinstance(dest_width, int):
        #     dest_width = dest_width.item()
        #     dest_height = dest_height.item()
        
        # box[:, 0] = np.clip(
        #     np.round(box[:, 0] / width * dest_width), 0, dest_width)
        # box[:, 1] = np.clip(
        #     np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes[index, :, :] = box.astype(np.int32)
    return boxes

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
            points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]