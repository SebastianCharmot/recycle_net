import os
import cv2
import json
import glob
import random
import detectron2
import numpy as np
import pandas as pd

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

setup_logger()

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def get_board_dicts(imgdir):
    json_file = imgdir+"/_annotations.coco.json" #Fetch the json file
    new_dict = []
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    for i in range(len(dataset_dicts["images"])):
        entry = {}
        entry["image_id"] = int(dataset_dicts["images"][i]["id"])
        entry["file_name"] = imgdir+"/"+dataset_dicts["images"][i]["file_name"]
        entry["height"] = dataset_dicts["images"][i]["height"]
        entry["width"] = dataset_dicts["images"][i]["width"]
        
        annots = []
        
        for j in range(len(dataset_dicts["annotations"])):
            annot = {}
            if (entry["image_id"] == int(dataset_dicts["annotations"][j]["image_id"])):
                annot["bbox"] = dataset_dicts["annotations"][j]["bbox"]
                annot["bbox_mode"] = BoxMode.XYWH_ABS
                annot["category_id"] = int(dataset_dicts["annotations"][j]["category_id"])
                annots.append(annot)
        
        entry["annotations"] = annots
        
        new_dict.append(entry)

    return new_dict

def run_experiment(model, dataset, generate_inference=False):
    try:
        for d in ["test","train", "valid"]:
            DatasetCatalog.register("boardetect_" + d + str(dataset), lambda d=d: get_board_dicts("DL_final_v"+ str(dataset) +"/" + d))
            MetadataCatalog.get("boardetect_" + d + str(dataset)).set(thing_classes=["garbage","glass","metal","paper","plastic"])

        board_metadata = MetadataCatalog.get("boardetect_train" + str(dataset))
    except: 
        pass

    cfg = get_cfg()

    if (model == 1):
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "model_final_68b088.pkl"
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "model_final_f6e8b1.pkl"

    cfg.DATASETS.TRAIN = ("boardetect_train" + str(dataset),)
    cfg.DATASETS.TEST = ("boardetect_valid" + str(dataset),)

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0125

    cfg.SOLVER.MAX_ITER = 2500

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("boardetect_test" + str(dataset), cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "boardetect_test" + str(dataset))
    inference_on_dataset(trainer.model, val_loader, evaluator)

    if (generate_inference):
        cfg.DATASETS.TEST = ("boardetect_test" + str(dataset), )
        predictor = DefaultPredictor(cfg)
        test_metadata = MetadataCatalog.get("boardetect_test" + str(dataset))

        counter = 0

        for imageName in glob.glob('DL_final_v' + str(dataset) + '/test/*jpg'):
            if (counter < 8):
                print(counter)
                im = cv2.imread(imageName)
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1],
                            metadata=test_metadata, 
                            scale=0.8
                             )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                plt.imshow(out.get_image()[:, :, ::-1])
                plt.show()
            counter += 1