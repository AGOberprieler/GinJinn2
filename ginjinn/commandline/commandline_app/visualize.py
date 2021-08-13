''' visualize command
'''

import os
import sys
from enum import Enum
import tqdm
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

def ginjinn_visualize(args):
    '''visualize

    GinJinn visualize command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn visualize.
    '''

    dataset_dir = args.dataset_dir
    image_dir = args.image_dir
    ann_path = args.ann_path
    input_type = process_input(image_dir, ann_path, dataset_dir)

    out_dir = args.out_dir
    vis_type = args.vis_type

    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_dataset_dir,
        check_ann_path,
        check_image_dir,
        AnnotationType,
        DatasetType,
        prepare_out_dir,
    )

    # Dataset input
    if input_type == InputType.Dataset:
        ds_type, ann_type, _ = check_dataset_dir(dataset_dir)
        if ds_type == DatasetType.split:
            print('ERROR: split datasets are not supported.')
            sys.exit(1)

        image_dir = os.path.join(dataset_dir, 'images')
        if ann_type == AnnotationType.COCO:
            ann_path = os.path.join(dataset_dir, 'annotations.json')
        elif ann_type == AnnotationType.COCO:
            ann_path = os.path.join(dataset_dir, 'annotations')
        if not out_dir:
            out_dir = os.path.join(dataset_dir, 'visualization')

    # Annotations and image input
    elif input_type == InputType.AnnotationsAndImages:
        check_image_dir(image_dir)
        ann_type = check_ann_path(ann_path)

        if not out_dir:
            print(
                'ERROR: -o/--out_dir is required when -a/--ann_path and -i/--image_dir '
                'are provided.'
            )
            sys.exit(1)

    # Should not get here
    else:
        print('ERROR: unknown input type.')
        sys.exit(1)

    if ann_type == AnnotationType.COCO:
        ann_type = 'COCO'
    elif ann_type == AnnotationType.PVOC:
        ann_type = 'PVOC'
    # Should not get here
    else:
        print('ERROR: unknown annotation type.')
        sys.exit(1)

    prepare_out_dir(out_dir)

    from ginjinn.utils import get_image_files
    from ginjinn.utils.utils import visualize_annotations
    n_images = len(get_image_files(image_dir))
    with tqdm.tqdm(total=n_images, desc='visualizing', unit='image') as pbar:
        visualize_annotations(
            ann_path=ann_path,
            img_dir=image_dir,
            out_dir=out_dir,
            ann_type=ann_type,
            vis_type=vis_type,
            progress_callback=pbar.update,
        )

    print(f'Visualizations written to "{out_dir}".')
