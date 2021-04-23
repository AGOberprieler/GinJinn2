''' Evaluation module
'''

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from ginjinn.ginjinn_config import GinjinnConfiguration
import os

def evaluate_detectron(
    cfg: CfgNode,
    task: str,
    dataset: str = "test",
    checkpoint_name: str = "model_final.pth",
):
    """Evaluate registered test dataset using COCOEvaluator

    Parameters
    ----------
    cfg : CfgNode
        Detectron2 configuration
    task : str
        "bbox-detection" or "instance-segmentation"
    dataset : str
        Name of registered dataset
    checkpoint_name : str
        Checkpoint name

    Returns
    -------
    eval_results : OrderedDict
        AP values
    """
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, checkpoint_name)

    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(
        model,
        save_dir = cfg.OUTPUT_DIR
    )

    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)


    if task == "bbox-detection":
        eval_tasks = ("bbox", )
    if task == "instance-segmentation":
        eval_tasks = ("bbox", "segm")

    evaluator = COCOEvaluator(
        dataset,
        tasks = eval_tasks,
        distributed = False,
        output_dir = cfg.OUTPUT_DIR
    )
    test_loader = build_detection_test_loader(cfg, dataset)
    eval_results = inference_on_dataset(model, test_loader, evaluator)
    return eval_results

def evaluate(
    cfg: GinjinnConfiguration,
    checkpoint_name: str = "model_final.pth",
):
    """Evaluate registered test dataset using COCOEvaluator

    Parameters
    ----------
    cfg : GinjinnConfiguration
        Ginjinn configuration object.
    checkpoint_name : str
        Checkpoint name.

    Returns
    -------
    eval_results : OrderedDict
        AP values
    """

    return evaluate_detectron(
        cfg.to_detectron2_config(is_test=True),
        task = cfg.task,
        dataset = "test",
        checkpoint_name=checkpoint_name,
    )
