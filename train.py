import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_brc_dicts(img_dir, label_dir):

    json_files = sorted(os.listdir(label_dir)) # change to right directory
    

    dataset_dicts = []
    for idx, json_file in enumerate(json_files):
        print(json_file)
        with open(os.path.join(label_dir, json_file)) as f:
            imgs_ann = json.load(f)
        record = {}
        
        filename = os.path.join(img_dir, imgs_ann["imagePath"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = imgs_ann["shapes"]
        objs = []
        for anno in annos:
            
            if anno["shape_type"] == 'polygon':
                poly = anno["points"]
                print('poly: ',poly)
                px = [p[0] for p in poly]
                py = [p[1] for p in poly]
                obj = {
                    "bbox": [min(px), min(py), max(px), max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
            elif anno["shape_type"] == 'rectangle':
                A, B = anno["points"]

                obj = {
                    "bbox": [A[0], A[1], B[0], B[1]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [[[A[0],A[1]],[B[0],A[1]],[B[0],B[1]],[A[0],B[1]]]],
                    "category_id": 0,
                }
            objs.append(obj)
            print('objs: ',objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

    

    
for d in ["train", "val"]:
    brc_dicts = get_brc_dicts(
        "datasets/train_val/images/" + d,
        "datasets/train_val/labels_object/" + d)
    print('brc_dicts: ',brc_dicts)
    with open("datasets/train_val/images/" + d + ".json", 'w') as fp:
        json.dump(brc_dicts, fp, indent = 4)
    def get_function(brc_dicts):
        return brc_dicts
    DatasetCatalog.register("brc_" + d, lambda d=d: get_function(brc_dicts))
    MetadataCatalog.get("brc_" + d).set(thing_classes=["brc"])
brc_metadata = MetadataCatalog.get("brc_train")


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("brc_train",)
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

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()