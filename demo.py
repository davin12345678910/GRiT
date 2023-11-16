import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo


# constants
WINDOW_NAME = "GRiT"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


'''''''''
Description: This will allow us to get the dense captions for each of the objects in a given image 
'''
if __name__ == "__main__":

    start_time = time.time()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # here we will be getting the image
    img = read_image("//home//davin123//GRiT//given_image.jpg", format="BGR")

    # here we will output the results onto the image
    predictions, visualized_output = demo.run_on_image(img)
    out_filename = args.output
    visualized_output.save(out_filename)

    # we will save the output from the model 
    instances = predictions["instances"]

    # get the bounding boxes of the dense captions
    bounding_boxes = instances.pred_boxes.tensor.tolist()

    # here we will get all of the dense captions 
    descriptions = instances.pred_object_descriptions
    descriptions = descriptions.data

    # here we will get all of the data which we will later 
    # store into a json 
    data = {}

    # here we will be storing the bounding boxes
    index = 0
    for bbox in bounding_boxes:
        data[index] = {"bbox" : bbox}
        index = index + 1

    # here we will be storing the dense captions 
    index = 0
    for description in descriptions:
        data[index]["label"] = description
        index = index + 1

    # Here we will be getting all of the objects into a list 
    grit_objects = []
    for current in data:
        grit_objects.append(data[current])

    # this is the final json object
    final_json = {"results" : grit_objects}

    # here we will write the output to a json file 
    with open("//home//davin123//GRiT//output.json", 'w') as json_file:
        json.dump(final_json, json_file, indent=4)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Totaled elapsed_time: ", elapsed_time)
