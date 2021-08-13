''' ginjinn utils count functionality
'''

from ginjinn.commandline.commandline_app.commandline_helpers import AnnotationType, DatasetType, check_ann_path, check_dataset_dir
import os
import sys
from enum import Enum

class InputType(Enum):
    '''InputType
    '''
    Dataset = 0
    Annotations = 1

def process_input(
    ann_path: str,
    dataset_dir: str,
) -> InputType:
    '''process_input

    Exits on invalid input.

    Parameters
    ----------
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
        if len(warn_msgs) > 0:
            print('\n'.join(warn_msgs))
        return InputType.Dataset

    elif ann_path is not None:
        return InputType.Annotations

    else:
        print(
            'ERROR: invalid input arguments. Provide either -I/--datset_dir, '
            'or -i/--image_dir and -a/--ann_path.'
        )
        sys.exit(1)

def utils_count(args):
    '''utils_count

    GinJinn utils count command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        count subcommand.
    '''

    out_file = args.out_file
    dataset_dir = args.dataset_dir
    ann_path = args.ann_path

    input_type = process_input(ann_path, dataset_dir)

    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_dataset_dir,
        check_ann_path,
        AnnotationType,
        DatasetType,
        prepare_out_file,
    )
    # Dataset input
    if input_type == InputType.Dataset:
        ds_type, ann_type, _ = check_dataset_dir(dataset_dir)
        if ds_type == DatasetType.split:
            print('ERROR: split datasets are not supported.')
            sys.exit(1)
        if ann_type == AnnotationType.COCO:
            ann_path = os.path.join(dataset_dir, 'annotations.json')
        else:
            ann_path = os.path.join(dataset_dir, 'annotations')
    # Annotations input
    else:
        ann_type = check_ann_path(ann_path)

    from ginjinn.utils.utils import count_categories, count_categories_pvoc
    if ann_type == AnnotationType.COCO:
        count_df = count_categories(ann_path)
    else:
        count_df = count_categories_pvoc(ann_path)

    prepare_out_file(out_file)
    count_df.to_csv(out_file, index_label='image')

    if args.show:
        import pandas as pd
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None
        ):
            print(count_df)

    print(f'\nCategory counts written to {args.out_file}.')
