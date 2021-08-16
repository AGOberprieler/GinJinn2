''' ginjinn utils crop functionality
'''

import sys
import os
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

def crop(
    ann_path: str,
    image_dir: str,
    out_dir: str,
    padding: int,
    cropping_type: str,
):
    '''crop

    Crop annotations. Can exit.

    Parameters
    ----------
    ann_path : str
        Annotation path.
    image_dir : str
        Image directory.
    out_dir : str
        Output directory.
    padding : int
        Padding.
    cropping_type : str
        Cropping type.
    '''
    from ginjinn.utils import crop_seg_from_coco
    from ginjinn.utils import crop_bbox_from_coco
    from ginjinn.commandline.commandline_app.commandline_helpers import (
        get_n_annotations
    )

    n_ann = get_n_annotations(ann_path)
    if cropping_type == 'segmentation':
        with tqdm.tqdm(total=n_ann, desc='cropping', unit='ann') as pbar:
            crop_seg_from_coco(
                ann_file=ann_path,
                img_dir=image_dir,
                outdir=out_dir,
                padding=padding,
                progress_callback=pbar.update,
            )
    elif cropping_type == 'bbox':
        with tqdm.tqdm(total=n_ann, desc='cropping', unit='ann') as pbar:
            crop_bbox_from_coco(
                ann_file=ann_path,
                img_dir=image_dir,
                outdir=out_dir,
                padding=padding,
                progress_callback=pbar.update,
            )
    else:
        print(f'ERROR: unknown cropping type "{cropping_type}"')
        sys.exit(1)

def utils_crop(args):
    '''utils_crop

    GinJinn utils crop command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils crop
        subcommand.
    '''

    from ginjinn.utils import crop_seg_from_coco
    from ginjinn.utils import crop_bbox_from_coco

    dataset_dir = args.dataset_dir
    image_dir = args.image_dir
    ann_path = args.ann_path
    input_type = process_input(image_dir, ann_path, dataset_dir)

    out_dir = args.out_dir
    padding = args.padding
    cropping_type = args.cropping_type

    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_dataset_dir,
        check_ann_path,
        check_image_dir,
        DatasetType,
        AnnotationType,
        prepare_out_dir,
        get_n_annotations,
    )

    # Dataset input
    if input_type == InputType.Dataset:
        ds_type, ann_type, splits = check_dataset_dir(dataset_dir)

        if ann_type == AnnotationType.PVOC:
            print('ERROR: PVOC datasets are not supported.')
            sys.exit(1)

        # Split dataset input
        if ds_type == DatasetType.split:
            prepare_out_dir(out_dir)
            for split in splits:
                split_dir = os.path.join(dataset_dir, split)
                ann_path = os.path.join(split_dir, 'annotations.json')
                image_dir = os.path.join(split_dir, 'images')
                split_out_dir = os.path.join(out_dir, split)
                print(f'Cropping dataset {split_dir}:')
                crop(
                    ann_path,
                    image_dir,
                    split_out_dir,
                    padding,
                    cropping_type,
                )
            print(
                f'Cropped split dataset written to "{out_dir}".'
            )

        # Simple dataset input
        else:
            ann_path = os.path.join(dataset_dir, 'annotations.json')
            image_dir = os.path.join(dataset_dir, 'images')

            prepare_out_dir(out_dir)
            crop(
                ann_path,
                image_dir,
                out_dir,
                padding,
                cropping_type,
            )
            print(
                f'Cropped dataset written to "{out_dir}".'
            )

    # Annotations and images input
    else:
        check_image_dir(image_dir)
        ann_type = check_ann_path(ann_path)
        if ann_type == AnnotationType.PVOC:
            print('ERROR: PVOC annotations are not supported.')
            sys.exit(1)

        prepare_out_dir(out_dir)
        crop(
            ann_path,
            image_dir,
            out_dir,
            padding,
            cropping_type,
        )
        print(
            f'Cropped dataset written to "{out_dir}".'
        )
