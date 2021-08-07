''' ginjinn utils sw_split functionality
'''

import warnings
import os
import sys
import shutil
from typing import Iterable, Tuple


def process_window_size(
    window_size: Iterable[int],
) -> Tuple[int, int]:
    '''process_window_size

    Parameters
    ----------
    window_size : Iterable[int]
        window_size command line argument(s)

    Returns
    -------
    Tuple[int, int]
        width, height
    '''
    if len(window_size) == 1:
        win_width = window_size[0]
        win_height = window_size[0]
    elif len(window_size) == 2:
        win_width = window_size[0]
        win_height = window_size[1]
    else:
        print('ERROR: "-s/--window_size" must receive 1 or 2 arguments.')
        sys.exit(1)

    return win_width, win_height

def process_overlap(
    overlap: Iterable[int],
) -> Tuple[int, int]:
    '''process_overlap

    Parameters
    ----------
    overlap : Iterable[int]
        overlap command line argument(s)

    Returns
    -------
    Tuple[int, int]
        horizontal, vertical
    '''
    if len(overlap) == 1:
        hor_overlap = overlap[0]
        vert_overlap = overlap[0]
    elif len(overlap) == 2:
        hor_overlap = overlap[0]
        vert_overlap = overlap[1]
    else:
        print('ERROR: "-p/--overlap" must receive 1 or 2 arguments.')
        sys.exit(1)

    return hor_overlap, vert_overlap

def utils_sw_split(args):
    '''utils_sw_split

    GinJinn utils sw_split command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sw_split subcommand.
    '''

    win_width, win_height = process_window_size(args.window_size)
    hor_overlap, vert_overlap = process_overlap(args.overlap)

    print(win_width, win_height)
    print(hor_overlap, vert_overlap)

    image_dir = args.image_dir
    ann_path = args.ann_path
    dataset_dir = args.dataset_dir

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
        print('Will use -I/--dataset_dir')

    elif ann_path is not None and image_dir is not None:
        print('Will use -a/--ann_path and -i/--image_dir')
    elif image_dir is not None:
        print('Will use -i/--image_dir')
    else:
        print(
            'ERROR: invalid input arguments. Provide either -I/--datset_dir, or -i/--image_dir, '
            'or -i/--image_dir AND -a/--ann_path.'
        )
        sys.exit(1)

    print('image_dir:', image_dir)
    print('ann_path:', ann_path)
    print('dataset_dir:', dataset_dir)

    from ginjinn.utils import confirmation_cancel

    # image splitting
    from ginjinn.utils.dataset_cropping import sliding_window_crop_images

    # dataset splitting
    from ginjinn.utils.dataset_cropping import sliding_window_crop_coco, sliding_window_crop_pvoc
