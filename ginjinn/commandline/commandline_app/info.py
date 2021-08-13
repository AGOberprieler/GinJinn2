''' Module for the ginjinn info subcommand
'''

import sys
import os
from enum import Enum

class InputType(Enum):
    '''InputType
    '''
    Dataset = 0
    AnnotationsAndImages = 1

def process_input(
    image_dir: str,
    ann_path: str,
    dataset_dir: str,
) -> InputType:
    '''process_input

    Exits on invalid input.

    Parameters
    ----------
    image_dir : str
        image_dir commandline argument
    ann_path : str
        ann_path commandline argument
    dataset_dir : str
        dataset_dir commandline argument

    Returns
    -------
    InputType
        Type of provided input
    '''

    if dataset_dir is not None:
        warn_msgs = []
        if ann_path is not None:
            warn_msgs.append(
                'WARNING: -I/--dataset_dir was provided. -a/--ann_path will be ignored.'
            )
        if image_dir is not None:
            warn_msgs.append(
                'WARNING: -I/--dataset_dir was provided. -i/--image_dir will be ignored.'
            )
        if len(warn_msgs) > 0:
            print('\n'.join(warn_msgs))
        return InputType.Dataset

    elif ann_path is not None and image_dir is not None:
        return InputType.AnnotationsAndImages

    else:
        print(
            'ERROR: invalid input arguments. Provide either -I/--datset_dir, '
            'or -i/--image_dir and -a/--ann_path.'
        )
        sys.exit(1)

def ginjinn_info(args):
    '''ginjinn_info

    GinJinn info command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn
        info subcommand.
    '''
    from ginjinn.utils.utils import dataset_info

    dataset_dir = args.dataset_dir
    ann_path = args.ann_path
    image_dir = args.image_dir

    input_type = process_input(image_dir=image_dir, ann_path=ann_path, dataset_dir=dataset_dir)

    # Dataset input
    if input_type == InputType.Dataset:
        from ginjinn.commandline.commandline_app.commandline_helpers import (
            check_dataset_dir,
            DatasetType,
            AnnotationType,
        )
        ds_type, ann_type, _ = check_dataset_dir(dataset_dir=dataset_dir)

        if ds_type == DatasetType.split:
            print(
                'ERROR: ginjinn info can only be applied to simple (non-split) datasets.'
            )
            sys.exit(1)

        if ann_type == AnnotationType.COCO:
            ann_path = os.path.join(dataset_dir, 'annotations.json')
        else:
            ann_path = os.path.join(dataset_dir, 'annotations')
        image_dir = os.path.join(dataset_dir, 'images')

        ann_type = 'COCO' if ann_type == AnnotationType.COCO else 'PVOC'

    # Annotations and images input
    else:
        from ginjinn.commandline.commandline_app.commandline_helpers import (
            check_ann_path,
            check_image_dir,
            AnnotationType,
        )

        ann_type = check_ann_path(ann_path)
        check_image_dir(image_dir)

        ann_type = 'COCO' if ann_type == AnnotationType.COCO else 'PVOC'

    dataset_info(
        ann_path=ann_path,
        img_dir=image_dir,
        ann_type=ann_type,
    )
