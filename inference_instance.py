import logging
import os
import random
from os.path import basename

import cv2
from detectron2.data import DatasetCatalog
from module.instance.predictor import DefaultPredictorWithProposal as DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer, GenericMask
from tqdm import tqdm

from common.cmd_parser import parse_cmd_arg
from common.utils import plt_show, join, mkdirs_if_not_exist
from initializer.instance_initializer import InferenceInstanceInitializer
from module.instance.evaluator import EnhancedCOCOEvaluator
from module.instance.trainer import TrainerWithoutHorizontalFlip
from pre_process.pre_process import read_to_gray_scale

from detectron2 import model_zoo

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from detectron2.data.datasets import register_coco_instances
from dataset.coco_format_instance import register_coco_format_instance
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
import torch
import numpy as np
from PIL import Image


def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


class OurVisualizer(Visualizer):

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        # boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        boxes = None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        labels = None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


def main(init: InferenceInstanceInitializer):
    # this need to set in config yaml file
    # path to the model we just trained
    # config.MODEL.WEIGHTS = join('output/debug/20210608.232202', "model_final.pth")
    # set a custom testing threshold
    # config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    logger = logging.getLogger('detectron2')

    config = init.config
    dataset_metadata = init.dataset_metadata
    dataset_metadata.thing_classes = ['wangou', 'na', 'ti', 'pie', 'piezhe', 'piedian', 'xiegouhuowogou', 'heng',
                                      'hengzhe', 'hengzhezhehuohengzhewan', 'hengzhezhezhe',
                                      'hengzhezhezhegouhuohengpiewangou', 'hengzhezhepie', 'hengzheti',
                                      'hengzhegou', 'hengpiehuohenggou', 'hengxiegou', 'dian', 'shu', 'shuwan',
                                      'shuwangou', 'shuzhezhegou', 'shuzhepiehuoshuzhezhe', 'shuti', 'shugou']

    predictor = DefaultPredictor(config)
    visualize_prediction_path = join(config.OUTPUT_DIR, 'visualize_prediction')
    visualize_proposal_path = join(config.OUTPUT_DIR, 'visualize_proposal')
    visualize_input_path = join(config.OUTPUT_DIR, 'visualize_input')

    mkdirs_if_not_exist(visualize_prediction_path)
    mkdirs_if_not_exist(visualize_proposal_path)
    mkdirs_if_not_exist(visualize_input_path)

    # predict and visualize the image provided in image paths
    if config.IMAGE_PATHS is not None:
        for image_path in tqdm(config.IMAGE_PATHS):
            im = read_to_gray_scale(image_path)
            plt_show(im[:, :, ::-1], save_filename=join(visualize_input_path, basename(image_path)))

            outputs = predictor(im)

            # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for
            # specification
            # print(outputs["instances"].pred_classes)
            # print(outputs["instances"].pred_boxes)

            # We can use `Visualizer` to draw the predictions on the image.
            v = OurVisualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt_show(out.get_image()[:, :, ::-1], save_filename=join(visualize_prediction_path, basename(image_path)))

            # We also visualize the proposal boxes
            v = OurVisualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
            box_size = min(len(outputs["proposals"].proposal_boxes), config.NUM_VIS_PROPOSAL)
            out = v.overlay_instances(boxes=outputs["proposals"].proposal_boxes[:box_size].to("cpu"))
            plt_show(out.get_image()[:, :, ::-1], save_filename=join(visualize_proposal_path, basename(image_path)))

    # evaluate the model and get bbox, segm metrics
    evaluate(init=init, config=config)

    config.VIS_DATASET_RESULT = True
    if config.VIS_DATASET_RESULT:
        # visualize prediction in dataset
        visualize_prediction_in_datasets(config=config,
                                         dataset_name=init.val_set_name,
                                         dataset_metadata=dataset_metadata,
                                         num_vis=None,
                                         logger=logger)


def evaluate(init, config):
    evaluator = EnhancedCOCOEvaluator(init.val_set_name, config.TASKS, False, output_dir=config.OUTPUT_DIR)

    trainer = TrainerWithoutHorizontalFlip(config)
    trainer.resume_or_load()
    trainer.test(config, model=trainer.model, evaluators=[evaluator])


def visualize_prediction(predictor, dataset_name, dataset_metadata, OUTPUT_DIR, logger, num_vis=10,
                         num_vis_proposal=20):
    visualize_prediction_path = join(OUTPUT_DIR, 'visualize_dataset_prediction')
    visualize_proposal_path = join(OUTPUT_DIR, 'visualize_proposal_prediction')
    logger.info("Saving prediction visualization results in {}".format(visualize_prediction_path))
    logger.info("Saving proposal visualization results in {}".format(visualize_proposal_path))
    if not os.path.exists(visualize_prediction_path):
        os.makedirs(visualize_prediction_path)
    if not os.path.exists(visualize_proposal_path):
        os.makedirs(visualize_proposal_path)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if num_vis is None:
        vis_collect = dataset_dicts
    else:
        vis_collect = random.sample(dataset_dicts, num_vis)
    for d in tqdm(vis_collect):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = OurVisualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=5.0,
                       # instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt_show(out.get_image()[:, :, ::-1], join(visualize_prediction_path,
                                                   os.path.basename(d['file_name'])))

        # We also visualize the proposal boxes
        v = OurVisualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=5.0)
        box_size = min(len(outputs["proposals"].proposal_boxes), num_vis_proposal)
        out = v.overlay_instances(boxes=outputs["proposals"].proposal_boxes[:box_size].to("cpu"))
        plt_show(out.get_image()[:, :, ::-1], save_filename=join(visualize_proposal_path,
                                                                 os.path.basename(d['file_name'])))


def visualize_prediction_in_datasets(config, dataset_name, dataset_metadata, logger, num_vis=10):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    config.MODEL.WEIGHTS = config.MODEL.WEIGHTS  # path to the model we just trained
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(config)
    visualize_prediction(predictor, dataset_name, dataset_metadata, config.OUTPUT_DIR, logger, num_vis=num_vis,
                         num_vis_proposal=config.NUM_VIS_PROPOSAL)


if __name__ == '__main__':
    args = parse_cmd_arg()

    initializer = InferenceInstanceInitializer(args.config)
    main(initializer)
