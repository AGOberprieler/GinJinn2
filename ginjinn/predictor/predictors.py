"""
Classes for prediction, output formatting, and visualization.
"""

import datetime
import glob
import json
import os
import pickle
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Optional, Union
import numpy as np
import cv2
import imantics
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import _create_text_labels, ColorMode, GenericMask
from detectron2.utils.visualizer import VisImage, Visualizer
import ginjinn.segmentation_refinement as refine
import torch
from ginjinn.data_reader.data_reader import get_class_names
from ginjinn.ginjinn_config import GinjinnConfiguration
from ginjinn.utils.utils import bbox_from_polygons


class GinjinnPredictor():
    '''A class for predicting from a trained Detectron2 model.

    Parameters
    ----------
    cfg : CfgNode
        Detectron2 configuration object
    class_names : list of str
        Ordered list of object class names
    img_dir : str
        Directory containing input images for inference
    outdir : str
        Directory for writing results
    task : str
        "bbox-detection" or "instance-segmentation"
    '''
    def __init__(
        self,
        cfg: CfgNode,
        class_names: List[str],
        img_dir: str,
        outdir: str,
        task: str
    ):
        self.class_names = class_names
        self.d2_cfg = cfg
        self.img_dir = img_dir
        self.outdir = outdir
        self.task = task
        self._coco_annotations = dict()
        self._coco_images = dict()

    @classmethod
    def from_ginjinn_config(
        cls,
        gj_cfg: GinjinnConfiguration,
        img_dir: str,
        outdir: str,
        checkpoint_name: str = "model_final.pth",
    ) -> "GinjinnPredictor":
        """
        Build GinjinnPredictor object from GinjinnConfiguration instead of
        Detectron2 configuration.

        Parameters
        ----------
        gj_cfg : GinjinnConfiguration
        img_dir : str
            Directory containing input images for inference
        outdir : str
            Directory for writing results
        checkpoint_name : str
            Name of the checkpoint to use.

        Returns
        -------
        GinjinnPredictor
        """

        d2_cfg = gj_cfg.to_detectron2_config()
        d2_cfg.MODEL.WEIGHTS = os.path.join(d2_cfg.OUTPUT_DIR, checkpoint_name)

        return cls(
            d2_cfg,
            get_class_names(gj_cfg.project_dir),
            img_dir,
            outdir,
            gj_cfg.task
        )


    def predict(
        self,
        img_names: List[str] = [],
        output_options: Iterable[str] = ("COCO", "cropped", "visualization"),
        padding: int = 0,
        threshold: Union[float, int] = 0.8,
        seg_refinement: bool = False,
        refinement_device: str = "cuda:0",
        refinement_method: str = "full"
    ):
        """
        img_names : list of str, default=[]
            File names of images to be used as input. By default, all images within self.img_dir
            will be used.
        output_options : iterable of str, default=("COCO", "cropped", "visualization")
            Available output formats:
                "COCO": Write predictions to COCO json file. For better compatibility with external
                        programs, the annotations do not contain polygons consisting of less than
                        three points (i.e., tiny sub-objects consisting of only one or two pixels).
                "cropped": Save cropped images and segmentation masks (if available).
                           In case of instance segmentation, an additional COCO json file with
                           annotations referring to the cropped images will be written.
                "visualization": Saves input images overlaid with object predictions.
        padding : int, default=0
            This option allows to increase the cropping range beyond the predicted bounding box.
            If possible, each side of the latter is shifted by the same number of pixels.
        threshold : float or int, default=0.8
            Minimum score of predicted instances
        seg_refinement : bool, default=False
            If true, predictions are postprocessed with CascadePSP.
            This option only works for instance segmentation.
        refinement_device : str, default="cuda:0"
            CPU or CUDA device for refinement with CascadePSP
        refinement_method : str, default="full"
            If set to "fast", the local refinement step will be skipped.
        """
        self.d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.d2_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold

        self._clear_coco("annotations")
        self._clear_coco("annotations_cropped")

        # create/clean subdirectories of self.outdir
        output_subdirs = []

        if "cropped" in output_options:
            output_subdirs.append("images_cropped")
            if self.task == "instance-segmentation":
                output_subdirs.append("masks_cropped")
        if "visualization" in output_options:
            output_subdirs.append("visualization")

        for subdir in output_subdirs:
            target_dir = os.path.join(self.outdir, subdir)
            os.makedirs(target_dir, exist_ok=True)
            for path in glob.iglob(os.path.join(target_dir, "*")):
                os.remove(path)

        # get image names
        if not img_names:
            patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
            img_paths = []
            for pat in patterns:
                img_paths.extend(glob.glob(os.path.join(self.img_dir, pat)))
            img_names = [os.path.split(path)[-1] for path in img_paths]

        # process images
        with NamedTemporaryFile() as tmpfile:

            d2_predictor = DefaultPredictor(self.d2_cfg)

            # save predictions to disc
            for i_img, img_name in enumerate(img_names):

                img_path = os.path.join(self.img_dir, img_name)
                image = cv2.imread(img_path)

                predictions = d2_predictor(image)

                # convert to numpy arrays
                boxes = predictions["instances"].get_fields()["pred_boxes"].to("cpu").tensor.numpy()
                classes = predictions["instances"].get_fields()["pred_classes"].to("cpu").numpy()
                scores = predictions["instances"].get_fields()["scores"].to("cpu").numpy()

                if self.task == "instance-segmentation":
                    masks = predictions["instances"].get_fields()["pred_masks"].to("cpu").numpy()
                else:
                    masks = None # np.array([])

                pickle.dump(boxes, tmpfile)
                pickle.dump(classes, tmpfile)
                pickle.dump(masks, tmpfile)
                pickle.dump(scores, tmpfile)

            d2_predictor = None
            torch.cuda.empty_cache()

            if self.task == "instance-segmentation" and seg_refinement:
                refiner = refine.Refiner(device=refinement_device)

            tmpfile.seek(0)

            # load predictions and apply refinement (optional)
            for i_img, img_name in enumerate(img_names):

                img_path = os.path.join(self.img_dir, img_name)
                image = cv2.imread(img_path)

                boxes = pickle.load(tmpfile)
                classes = pickle.load(tmpfile)
                masks = pickle.load(tmpfile)
                scores = pickle.load(tmpfile)

                if seg_refinement:
                    for i_mask, mask in enumerate(masks):
                        masks[i_mask] = refiner.refine(
                            image,
                            mask.astype("int") * 255,
                            fast = True if refinement_method=="fast" else False
                        )

                if self.task == "instance-segmentation":
                    for i_mask, mask in enumerate(masks):
                        # recalculate bounding boxes
                        x_any = masks[i_mask].any(axis=0)
                        y_any = masks[i_mask].any(axis=1)
                        x = np.where(x_any == True)[0]
                        y = np.where(y_any == True)[0]
                        if len(x) > 0 and len(y) > 0:
                            boxes[i_mask] = [x[0], y[0], x[-1] + 1, y[-1] + 1]
                        else:
                            boxes[i_mask] = [0, 0, 0, 0]

                if "COCO" in output_options:
                    self._update_coco("annotations", image, img_name, i_img + 1, boxes, classes, masks)

                if "cropped" in output_options:
                    self._save_cropped(image, img_name, i_img + 1, boxes, classes, masks, padding)

                if "visualization" in output_options:
                    self._save_visualization(image, img_name, scores, classes, boxes, masks)

            if "COCO" in output_options:
                self._save_coco("annotations")
                self._clear_coco("annotations")

                if self.task == "instance-segmentation":
                    self._save_coco("annotations_cropped")
                    self._clear_coco("annotations_cropped")

            refiner = None
            torch.cuda.empty_cache()


    def _save_visualization(
        self,
        image: np.ndarray,
        img_name: str,
        scores: np.ndarray,
        classes: np.ndarray,
        boxes: np.ndarray,
        masks: Optional[np.ndarray] = None
    ):

        # not existing before
        MetadataCatalog.get("pred").thing_classes = self.class_names

        visualizer = GJ_Visualizer(
            image[:, :, ::-1],
            metadata = MetadataCatalog.get("pred"),
            instance_mode = ColorMode.IMAGE_BW
        )
        vis_image = visualizer.draw_instance_predictions_gj(scores, classes, boxes, self.class_names, masks)

        outpath = os.path.join(
            self.outdir,
            "visualization",
            img_name
        )
        vis_image.save(outpath)


    def _save_cropped(
        self,
        image: np.ndarray,
        img_name: str,
        img_id: int,
        boxes: np.ndarray,
        classes: List[str],
        masks: Optional[np.ndarray] = None,
        padding: int = 5
    ):

        height, width = image.shape[:2]

        for i_inst, bbox in enumerate(boxes):

            x1, y1, x2, y2 = [round(coord) for coord in bbox]
            x1, x2 = np.clip((x1 - padding, x2 + padding), 0, width - 1)
            y1, y2 = np.clip((y1 - padding, y2 + padding), 0, height - 1)
            image_cropped = image[y1:y2, x1:x2]

            outpath = os.path.join(
                self.outdir,
                "images_cropped",
                "{}_{}.jpg".format(os.path.splitext(img_name)[0], i_inst + 1)
            )
            cv2.imwrite(outpath, image_cropped)

            if self.task == "instance-segmentation":
                mask_cropped = masks[i_inst][y1:y2, x1:x2]
                outpath = os.path.join(
                    self.outdir,
                    "masks_cropped",
                    "{}_{}.png".format(os.path.splitext(img_name)[0], i_inst + 1)
                )
                cv2.imwrite(outpath, mask_cropped.astype("uint8") * 255)

                # prepare COCO annotation of cropped image:

                # calculate bounding box
                x_any = mask_cropped.any(axis=0)
                y_any = mask_cropped.any(axis=1)
                x = np.where(x_any == True)[0]
                y = np.where(y_any == True)[0]
                if len(x) > 0 and len(y) > 0:
                    box = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1])
                else:
                    box = np.array([0, 0, 0, 0])

                self._update_coco(
                    "annotations_cropped",
                    image_cropped,
                    "img_{}_{}.jpg".format(img_id, i_inst + 1),
                    len(self._coco_images["annotations_cropped"]) + 1,
                    np.expand_dims(box, axis=0),
                    np.expand_dims(classes[i_inst], axis=0),
                    np.expand_dims(mask_cropped, axis=0)
                )


    def _update_coco(
        self,
        name: str,
        image: np.ndarray,
        img_name: str,
        img_id: int,
        boxes: np.ndarray,
        classes: List[str],
        masks: Optional[np.ndarray] = None
    ):

        height, width = image.shape[:2]

        self._coco_images[name].append({
            "id": img_id,
            "file_name": img_name,
            "coco_url": "",
            "date_captured": "",
            "flickr_url": "",
            "height": height,
            "width": width,
            "license": 1
        })

        for i_inst, bbox in enumerate(boxes):
            bbox = bbox.tolist()
            bbox_coco = [ bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1] ]

            anno = {
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "bbox": bbox_coco,
                "iscrowd": 0,
                "image_id": img_id,
                "id": len(self._coco_annotations[name]) + 1,
                "category_id": classes[i_inst].tolist() + 1
            }

            if self.task == "instance-segmentation":
                imask = imantics.Mask(masks[i_inst])
                ipoly = imask.polygons()
                # remove polygons with less than 3 points
                anno["segmentation"] = [p for p in ipoly.segmentation if len(p) >= 6]
                # recalculate bounding box
                anno["bbox"] = bbox_from_polygons(anno["segmentation"], fmt="xywh").tolist()
                anno["area"] = anno["bbox"][2] * anno["bbox"][3]

            self._coco_annotations[name].append(anno)


    def _save_coco(self, name: str):
        info = {
            "contributor" : "",
            "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
            "description" : "",
            "version" : "",
            "url" : "",
            "year" : ""
        }
        licenses = [{"id": 1, "name": "", "url": ""}]
        categories = [
            {"id": i+1, "name": cl, "supercategory": ""} for (i, cl) in enumerate(self.class_names)
        ]

        # write COCO annotation file
        json_new = os.path.join(self.outdir, name + ".json")
        with open(json_new, 'w') as json_file:
            json.dump({
                'info': info,
                'licenses': licenses,
                'images': self._coco_images[name],
                'annotations': self._coco_annotations[name],
                'categories': categories
                },
                json_file,
                indent=2,
                sort_keys=True
            )


    def _clear_coco(self, name: str):
        self._coco_annotations[name] = []
        self._coco_images[name] = []


class GJ_Visualizer(Visualizer):
    """Modified version of Detectron2's Visualizer class.
    """

    def draw_instance_predictions_gj(
        self,
        scores: np.ndarray,
        classes: np.ndarray,
        boxes: np.ndarray,
        class_names: List[str],
        masks: Optional[np.ndarray] = None,
        alpha: Union[int, float] = 0.2
    ) -> VisImage:
        """
        Modification of Detectron2's draw_instance_predictions() using differently formatted inputs.

        Parameters
        ----------
        scores : np.ndarray
            Scores of predictions
        classes : np.ndarray
            Predicted classes
        boxes : np.ndarray
            Predicted bounding boxes as 3D array
        class_names : list of str
            Ordered list of object class names
        masks : np.ndarray
            Predicted segmentation masks as 3D array
        alpha : float or int
            Opacity of color mask applied to each segmentation instance,
            should be within 0 and 1

        Returns
        -------
        VisImage
            Image object with visualizations
        """
        labels = _create_text_labels(classes, scores, class_names)

        colors = None

        if masks is not None:
            img_bw = self.img.astype("f4").mean(axis=2)
            img_bw = np.stack([img_bw] * 3, axis=2)
            img_bw = img_bw.round().astype("int")
            img_bw[masks.any(axis=0) > 0] = self.img[masks.any(axis=0) > 0]
            self.output = VisImage(img_bw)

        self.overlay_instances(
            masks = None if masks is None
                else [GenericMask(x, self.output.height, self.output.width) for x in masks],
            boxes = boxes,
            labels = labels,
            keypoints = None,
            assigned_colors = colors,
            alpha = alpha,
        )
        return self.output
