''' GinJinn split commandline module
'''

import pandas as pd
import sys

from ginjinn.utils import confirmation_cancel

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

    # print('Running ginjinn split')
    # print(args)

    # import here to avoid long startup time, when ginjinn_split is not called
    from ginjinn.data_reader.data_splitter import create_split_2
    from ginjinn.utils.utils import get_anntype, find_img_dir, ImageDirNotFound

    ann_path = args.annotation_path
    img_dir = args.image_dir
    split_dir = args.output_dir
    task = args.task
    ann_type = args.ann_type

    if not img_dir:
        try:
            img_dir = find_img_dir(ann_path)
        except ImageDirNotFound:
            print(
                f'ERROR: could not find "images" folder as sibling of "{ann_path}". Make sure ' +\
                f'there is an "images" folder in the same directory as "{ann_path}" or ' +\
                'explicitly pass "--image_dir".'
            )
            sys.exit()

    if ann_type == 'auto':
        ann_type = get_anntype(ann_path)

    p_val = args.validation_fraction
    p_test = args.test_fraction

    if create_split_2(
        ann_path=ann_path,
        img_path=img_dir,
        split_dir=split_dir,
        task=task,
        ann_type=ann_type,
        p_val=p_val,
        p_test=p_test,
        on_split_dir_exists=on_split_dir_exists,
        on_split_proposal=on_split_proposal,
        on_no_valid_split=on_no_valid_split,
    ):
        print(f'Datasets written to "{split_dir}".')
