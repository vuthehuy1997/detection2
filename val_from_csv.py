import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

with open('datasets/train_val/images/val.json', 'r') as f:
    brc_dicts = json.loads(f.read())
def get_function(brc_dicts):
    return brc_dicts
DatasetCatalog.register("brc_val", lambda d: get_function(brc_dicts))
MetadataCatalog.get("brc_val").set(thing_classes=["brc"])

brc_metadata = MetadataCatalog.get("brc_val")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("brc_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.



# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
from boxes_from_bitmap import boxes_from_bitmap

total_time = 0

for d in brc_dicts:    
    im = cv2.imread(d["file_name"])
    start = time.time()
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    total_time += (time.time() - start)
    v = Visualizer(im[:, :, ::-1],
                   metadata=brc_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    print(outputs["instances"].get("pred_masks"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'val',os.path.basename(d["file_name"])),out.get_image()[:, :, ::-1])

    # boxes = boxes_from_bitmap(outputs["instances"].get("pred_masks"), im.shape[1], im.shape[0])
    # # print('boxes: ', boxes)
    # for box in boxes:
    #     pts = box.reshape((-1,1,2))
    #     # print('pts: ', pts)
    #     im = cv2.polylines(im, [pts], True, (0,255,255), 2)
    # cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'val_box',os.path.basename(d["file_name"])), im)

    for idx, mask in enumerate(outputs["instances"].get("pred_masks")):
        
        mask_img = mask.cpu().numpy()*255
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'val',
            os.path.splitext(os.path.basename(d["file_name"]))[0] + '_mask' + str(idx) + '.jpg'), mask_img)
        
        # _bitmap = np.expand_dims(mask.cpu().numpy(), axis=0)
        # bitmap = _bitmap[0]
        # height, width = bitmap.shape
        # contours, _ = cv2.findContours(
        #     (bitmap*255).astype(np.uint8),
        #     cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(height, width)
        # pts = contours[0].reshape((-1,1,2))
        # print('pts: ', pts)
        # box_im = cv2.polylines(im, [pts], True, (0,255,255), 2)
        # cv2.imwrite(os.path.join('/data/huyvt/ocr/detectron2/output/val_box',
        #     os.path.splitext(os.path.basename(d["file_name"]))[0] + '_contours' + str(idx) + '.jpg'), box_im)

        # print('img: ',im.shape[1], im.shape[0])
        box = boxes_from_bitmap(mask, im.shape[1], im.shape[0])
        pts = box.reshape((-1,1,2))
        print('pts: ', pts)
        box_im = cv2.polylines(im, [pts], True, (0,255,255), 2)
        # cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'val_box',
        #     os.path.splitext(os.path.basename(d["file_name"]))[0] + '_box' + str(idx) + '.jpg'), box_im)

    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'val',
        os.path.splitext(os.path.basename(d["file_name"]))[0] + '_box' + '.jpg'), box_im)

print(total_time)
print(len(brc_dicts))


# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("cavet_val", output_dir="./output")
# val_loader = build_detection_test_loader(cfg, "cavet_val")
# print(inference_on_dataset(predictor.model, val_loader, evaluator))