import cv2
import numpy as np

im = cv2.imread('/data/huyvt/ocr/corner_detector/datasets/corner_dataset/images/val/stage0_00002.jpg')
print('img hsape: ', im.shape)
boxes = np.array([
    [[118,131],[218,131],[218,228],[118,228]],
    [[0,0],[300,0],[300,228],[0,228]]
])
for box in boxes:
    pts = box.reshape((-1,1,2))
    print('pts: ', pts)
    im = cv2.polylines(im, [pts], True, (0,255,255), 2)
cv2.imwrite('test.jpg', im)