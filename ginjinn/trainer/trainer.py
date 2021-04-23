# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by AGOberprieler, 2020, using modifications made by Marcelo Ortega:
# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

"""
Classes for training and, optionally, simultaneous validation.
"""

import copy
import datetime
import logging
import os
import time
import json
import re
from typing import List, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import numpy as np
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.config import CfgNode
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm
from ginjinn.ginjinn_config import GinjinnConfiguration

class Trainer(DefaultTrainer):
    """Trainer class which allows to set the applied augmentations at runtime.
    """
    _augmentations = []

    @classmethod
    def set_augmentations(cls, augmentations):
        """Specify augmentations for training.

        Parameters
        ----------
        augmentations : list
            Augmentations to be applied
        """
        cls._augmentations = augmentations

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        """Build data loader for training.

        Parameters
        ----------
        cfg : CfgNode
            Detectron2 config.

        Returns
        ----------
        torch.utils.data.DataLoader
            Data loader
        """
        augs = [T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )]
        augs.extend(cls._augmentations)

        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=augs
            )
        )

    @classmethod
    def from_ginjinn_config(cls, gj_cfg : GinjinnConfiguration) -> "Trainer":
        '''from_ginjinn_config

        Build Trainer object from GinjinnConfiguration instead of
        detectron2 configuration.

        Parameters
        ----------
        gj_cfg : GinjinnConfiguration
            GinjinnConfiguration object.

        Returns
        -------
        Trainer
            Trainer object
        '''

        cls.set_augmentations(gj_cfg.augmentation.to_detectron2_augmentations())

        detectron2_cfg = gj_cfg.to_detectron2_config()

        return cls(detectron2_cfg)

    ##alternative:
    #@classmethod
    #def build_train_loader(cls, cfg):
        #"""Build data loader for training.

        #Returns
        #----------
        #torch.utils.data.DataLoader
            #Data loader
        #"""
        #augs = [T.ResizeShortestEdge(
            #cfg.INPUT.MIN_SIZE_TRAIN,
            #cfg.INPUT.MAX_SIZE_TRAIN,
            #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        #)]
        #augs.extend(cls._augmentations)

        #return build_detection_train_loader(
            #cfg,
            #mapper = lambda data_dict: mapper_train(data_dict, augs)
        #)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(PlottingHook(self.cfg.TEST.EVAL_PERIOD, self.cfg.OUTPUT_DIR))
        return hooks


class ValTrainer(Trainer):
    """This trainer class evaluates validation data during training.
    """
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str) -> COCOEvaluator:
        """Builds COCO evaluator for a given dataset.

        Parameters
        ----------
        cfg : CfgNode
            Detectron2 config.
        dataset_name : str
            Name of the evaluation data set.

        Returns
        ----------
        COCOEvaluator
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-2, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(
                    self.cfg,
                    is_train=True, # required to obtain losses
                    # no flip
                    augmentations=[T.ResizeShortestEdge(
                        self.cfg.INPUT.MIN_SIZE_TRAIN,
                        self.cfg.INPUT.MAX_SIZE_TRAIN,
                        self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                    )]
                )
            )
        ))
        hooks.append(PlottingHook(self.cfg.TEST.EVAL_PERIOD, self.cfg.OUTPUT_DIR))
        return hooks


def mapper_train(dataset_dict: dict, augmentations: list):
    """
    This basic mapper function takes a dataset dictionary in Detectron2 format,
    and maps it to a format used by the model.

    Parameters
    ----------
    dataset_dict : dict
        Annotations for one image in Detectron2 format
    augmentations : list
        Augmentations and transformations to be applied

    Returns
    -------
    dict
        Format accepted by builtin models in Detectron2
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    detection_utils.check_image_size(dataset_dict, image)

    image, transforms = T.apply_transform_gens(augmentations, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # pylint: disable=E1101

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


class LossEvalHook(HookBase):
    # pylint: disable=E1101
    """
    This hook allows periodic loss calculation for the validation data set.
    It is executed every ``eval_period`` iterations and after the last iteration.

    Parameters
    ----------
    eval_period : int
        Period to calculate losses. If set to 0, they are only calculated after the
        last iteration.
    model : torch.nn.Module
        Model to be used
    data_loader : iterable
        produces data to be run by `model(data)`
    """
    def __init__(self, eval_period: int, model: torch.nn.Module, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # see evaluator.py from Detectron2
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses_all = {}
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            losses_batch = self._get_loss(inputs)
            if losses_all:
                for key in losses_batch:
                    losses_all[key].append(losses_batch[key])
            else:
                for key in losses_batch:
                    losses_all[key] = [losses_batch[key]]

        losses_mean = {key + "_val": np.mean(values) for (key, values) in losses_all.items()}
        losses_mean["total_loss_val"] = sum(losses_mean.values())
        self.trainer.storage.put_scalars(**losses_mean, smoothing_hint=False)

        comm.synchronize()
        return losses_mean

    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return metrics_dict

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class PlottingHook(HookBase):
    """
    This hook provides periodic plotting of losses and evaluation scores.
    It is executed every ``eval_period`` iterations and after the last iteration.

    Parameters
    ----------
    eval_period : int
        Period to plot losses and evaluation scores. If set to 0, they are only calculated
        after the last iteration.
    outdir : str
        Output directory
    """
    def __init__(self, eval_period: int, outdir: str):
        self.period = eval_period
        self.outdir = outdir
        self.metrics_df = None
        self.pp = None

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self._plot_all()

    def _plot_all(self):
        """Plot all metrics logged in metrics.json into metrics.pdf.
        """
        json_file = os.path.join(self.outdir, "metrics.json")
        if not os.path.isfile(json_file):
            return

        entries = []
        with open(json_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entries.append(entry)

        self.metrics_df = pd.DataFrame.from_records(entries).sort_values(by='iteration')
        colnames = self.metrics_df.columns.tolist()

        plt.ioff()
        self.pp = PdfPages(os.path.join(self.outdir, "metrics.pdf"))

        # plot losses
        p = re.compile(".*loss.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_losses(cols_sel, smooth=True)

        # plot evaluation scores (bbox)
        p = re.compile("^bbox/.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_metrics(
            cols_sel,
            nrow_grid = 2,
            ncol_grid = 3,
            dataset_name = "val",
            legend_pos = "lower right"
        )

        # plot evaluation scores (segmentation)
        p = re.compile("^segm/.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_metrics(
            cols_sel,
            nrow_grid = 2,
            ncol_grid = 3,
            dataset_name = "val",
            legend_pos = "lower right"
        )

        # plot remaining metrics
        self._plot_metrics(colnames, nrow_grid=3, ncol_grid=4)

        self.pp.close()

    def _plot_metrics(
        self,
        cols: List[str],
        nrow_grid: int = 2,
        ncol_grid: int = 3,
        width: Union[float, int] = 11.69,
        height: Union[float, int] = 8.27,
        dataset_name: str = None,
        legend_pos: str = "best"
    ):
        """Plot multiple metrics, arranged by a specfied grid.
        If the grid size is not sufficient to accomodate all subplots, additional pages
        are appended to the resulting PDF.

        Parameters
        ----------
        cols : list of str
            Names of data frame columns to be plotted
        nrow_grid : int
            Number of rows per PDF page
        ncol_grid : int
            Number of columns per PDF page
        width : float or int
            Page width in inches, defaults to A4 (landscape)
        height : float or int
            Page height in inches, defaults to A4 (landscape)
        dataset_name : str
            If specified, this is used as legend text.
        legend_pos : str
            Legend position, see matplotlib.pyplot.legend,
            e.g., "upper right", "lower left", "best", etc.
        """
        if not cols:
            return
        grid_size = nrow_grid * ncol_grid
        for i in range(0, len(cols), grid_size):
            cols_chunk = cols[i:i+grid_size]
            fig, axs = plt.subplots(nrow_grid, ncol_grid)
            fig.set_size_inches(width, height)
            for ax, metric_name in zip(axs.flat, cols_chunk):
                idcs = ~self.metrics_df[metric_name].isna()
                ax.plot(
                    self.metrics_df['iteration'][idcs],
                    self.metrics_df[metric_name][idcs],
                    label = dataset_name
                )
                ax.set_title(metric_name)
                if dataset_name:
                    ax.legend(loc=legend_pos)
            for ax in axs.flat:
                if not ax.lines:
                    ax.axis("off")
            fig.tight_layout()
            self.pp.savefig()

    def _plot_losses(
        self,
        cols: List[str],
        nrow_grid: int = 2,
        ncol_grid: int = 3,
        width: Union[float, int] = 11.69,
        height: Union[float, int] = 8.27,
        legend_pos: str = "best",
        smooth: bool = False,
        window_size: int = 10
    ):
        """Plot multiple losses, arranged by a specfied grid.
        In case a validation data set is used, its scores are combined with those of
        the training data set. If the grid size is not sufficient to accomodate all
        subplots, additional pages are appended to the resulting PDF.

        Parameters
        ----------
        cols : list of str
            Names of data frame columns (losses) to be plotted
        nrow_grid : int
            Number of rows per PDF page
        ncol_grid : int
            Number of columns per PDF page
        width : float or int
            Page width in inches, defaults to A4 (landscape)
        height : float or int
            Page height in inches, defaults to A4 (landscape)
        legend_pos : str
            Legend position, see matplotlib.pyplot.legend,
            e.g., "upper right", "lower left", "best", etc.
        smooth : bool
            If set to True, training losses are overlaid with their rolling mean.
        window_size : int
            Number of values to be averaged if smooth is set to True.
        """
        if not cols:
            return
        p = re.compile("^.*_val$")
        cols_train = [c for c in cols if not p.match(c)]
        grid_size = nrow_grid * ncol_grid
        for i in range(0, len(cols_train), grid_size):
            cols_chunk = cols_train[i:i+grid_size]
            fig, axs = plt.subplots(nrow_grid, ncol_grid)
            fig.set_size_inches(width, height)
            for ax, metric_name in zip(axs.flat, cols_chunk):
                idcs = ~self.metrics_df[metric_name].isna()
                if smooth:
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs],
                        label="train",
                        color="tab:blue",
                        alpha=0.3
                    )
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs].rolling(window_size).mean(),
                        label="train (smoothed)",
                        color="tab:blue"
                    )
                else:
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs],
                        label="train",
                        color="tab:blue"
                    )
                if metric_name + "_val" in cols:
                    idcs = ~self.metrics_df[metric_name + "_val"].isna()
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name + "_val"][idcs],
                        label="val",
                        color="tab:orange"
                    )
                ax.set_title(metric_name)
                ax.legend(loc=legend_pos, fontsize="small")
            for ax in axs.flat:
                if not ax.lines:
                    ax.axis("off")
            fig.tight_layout()
            self.pp.savefig()
