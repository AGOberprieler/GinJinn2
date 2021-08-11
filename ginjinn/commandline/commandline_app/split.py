''' GinJinn split commandline module
'''

from enum import Enum
import sys
import os

import pandas as pd

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

def on_split_dir_exists(split_dir: str) -> bool:
    '''on_split_dir_exists

    Callback for when the split output directory already exists.

    Parameters
    ----------
    split_dir : str
        Path to split directory.

    Returns
    -------
    bool
        Whether the existing directory should be overwritten.
    '''
    from ginjinn.utils import confirmation_cancel
    return confirmation_cancel(
        '"' + split_dir + '" already exists.\nDo you want do overwrite it? '+\
        'ATTENTION: This will DELETE "' + split_dir + '" and all subdirectories.\n'
    )

def on_split_proposal(split_df: 'pd.DataFrame') -> bool:
    '''on_split_proposal

    Callback for proposing a split.

    Parameters
    ----------
    split_df : 'pd.DataFrame'
        pandas.DataFrame containing split information.

    Returns
    -------
    bool
        Whether the proposed split should be accepted.
    '''
    from ginjinn.utils import confirmation_cancel
    df_pretty = pd.DataFrame(
        [[f'{a} ({round(b, 2)})' for a,b in zip(r, r / r.sum())] for _, r in split_df.iterrows()],
        columns=split_df.columns,
        index=split_df.index
    )

    print('\nSplit proposal:')
    print(df_pretty)
    return confirmation_cancel(
        '\nDo you want to accept this split? (Otherwise a new one will be generated.)\n'
    )

def on_no_valid_split() -> bool:
    '''on_no_valid_split

    Callback for when no valid split was found.


    Returns
    -------
    bool
        Whether another try for finding a valid split should be made.
    '''
    from ginjinn.utils import confirmation_cancel
    return confirmation_cancel(
        'Could not find a valid split. Try again?\n'
    )


def ginjinn_split(args):
    '''ginjinn_split

    GinJinn split command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn split
        subcommand.
    '''
    ann_path = args.ann_path
    image_dir = args.image_dir
    dataset_dir = args.dataset_dir
    out_dir = args.out_dir
    task = args.task

    p_val = args.validation_proportion
    p_test = args.test_proportion

    input_type = process_input(image_dir=image_dir, ann_path=ann_path, dataset_dir=dataset_dir)

    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_ann_path,
        check_image_dir,
        check_dataset_dir,
        AnnotationType,
        DatasetType,
        prepare_out_dir,
    )

    # Dataset input
    if input_type == InputType.Dataset:
        ds_type, ann_type, _ = check_dataset_dir(dataset_dir=dataset_dir)

        if ds_type == DatasetType.split:
            print(
                'ERROR: ginjinn split can only be applied to simple (non-split) datasets.'
            )
            sys.exit(1)

        if ann_type == AnnotationType.COCO:
            ann_path = os.path.join(dataset_dir, 'annotations.json')
        else:
            ann_path = os.path.join(dataset_dir, 'annotations')
        image_dir = os.path.join(dataset_dir, 'images')

        ann_type = 'COCO' if ann_type == AnnotationType.COCO else 'PVOC'
    # Annotation and images input
    else:
        ann_type = check_ann_path(ann_path)
        check_image_dir(image_dir)

        ann_type = 'COCO' if ann_type == AnnotationType.COCO else 'PVOC'
    
    prepare_out_dir(out_dir)

    from ginjinn.data_reader.data_splitter import create_split_2
    if create_split_2(
        ann_path=ann_path,
        img_path=image_dir,
        split_dir=out_dir,
        task=task,
        ann_type=ann_type,
        p_val=p_val,
        p_test=p_test,
        on_split_dir_exists=lambda _: True,
        on_split_proposal=on_split_proposal,
        on_no_valid_split=on_no_valid_split,
    ):
        print(f'Split datasets written to "{out_dir}".')
