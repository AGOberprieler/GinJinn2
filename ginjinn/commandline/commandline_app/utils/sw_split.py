''' ginjinn utils sw_split functionality
'''


import tqdm

from enum import Enum
import os
import sys
from typing import Iterable, Tuple


def process_window_size(
    window_size: Iterable[int],
) -> Tuple[int, int]:
    '''process_window_size

    Exits on invalid input.

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

    Exits on invalid input.

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

class InputType(Enum):
    '''InputType
    '''
    Images = 0
    Dataset = 1
    AnnotationsAndImages = 2

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
        # print('Will use -I/--dataset_dir')
        return InputType.Dataset

    elif ann_path is not None and image_dir is not None:
        # print('Will use -a/--ann_path and -i/--image_dir')
        return InputType.AnnotationsAndImages

    elif image_dir is not None:
        # print('Will use -i/--image_dir')
        return InputType.Images

    else:
        print(
            'ERROR: invalid input arguments. Provide either -I/--datset_dir, or -i/--image_dir, '
            'or -i/--image_dir AND -a/--ann_path.'
        )
        sys.exit(1)

def sw_split_images(
    image_dir: str,
    out_dir: str,
    win_width: int,
    win_height: int,
    hor_overlap: int,
    vert_overlap: int,
):
    '''sw_split_images

    Sliding window split images in image_dir.

    Parameters
    ----------
    image_dir: str
        Image directory
    out_dir : str
        Output directory
    win_width : int
        Window width
    win_height : int
        Window height
    hor_overlap : int
        Horizontal window overlap
    vert_overlap : int
        Vertical window overlap
    '''
    from ginjinn.utils.dataset_cropping import sliding_window_crop_images
    from ginjinn.utils import get_image_files
    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_image_dir
    )

    check_image_dir(image_dir)

    print(f'Splitting images from "{image_dir}" ...')

    with tqdm.tqdm(total=len(get_image_files(image_dir))) as pbar:
        sliding_window_crop_images(
            img_dir=image_dir,
            img_dir_out=out_dir,
            win_width=win_width,
            win_height=win_height,
            hor_overlap=hor_overlap,
            vert_overlap=vert_overlap,
            progress_callback=pbar.update,
        )
    print(f'Sliding window images written to "{out_dir}".')

def split_images_and_annotations(
    image_dir: str,
    ann_path: str,
    out_dir: str,
    win_width: int,
    win_height: int,
    hor_overlap: int,
    vert_overlap: int,
    img_id: int,
    obj_id: int,
    remove_empty: bool,
    remove_incomplete: bool,
    task: str,
):
    '''split_images_and_annotations

    Split images and annotations into sliding windows.

    Parameters
    ----------
    image_dir : str
        Image directory.
    ann_path : str
        Annotation path.
    out_dir : str
        Output directory
    win_width : int
        Width of sliding windows.
    win_height : int
        Height of sliding windows.
    hor_overlap : int
        Horizontal window overlap.
    vert_overlap : int
        Vertical window overlap.
    img_id : int
        Start image id for COCO sliding window images.
    obj_id : int
        Start object id for COCO sliding window images.
    remove_empty : bool
        Remove empty images.
    remove_incomplete : bool
        Remove incomplete (trimmed) object annotations.
    task : str
        Detection task.
    '''
    from ginjinn.utils import get_image_files
    from ginjinn.commandline.commandline_app.commandline_helpers import (
        check_ann_path,
        check_image_dir,
        AnnotationType,
    )
    ann_type = check_ann_path(ann_path)
    check_image_dir(image_dir)

    os.mkdir(os.path.join(out_dir, 'images'))

    if ann_type == AnnotationType.COCO:
        from ginjinn.utils.dataset_cropping import sliding_window_crop_coco
        with tqdm.tqdm(total=len(get_image_files(image_dir))) as pbar:
            sliding_window_crop_coco(
                img_dir=image_dir,
                ann_path=ann_path,
                img_dir_out=os.path.join(out_dir, 'images'),
                ann_path_out=os.path.join(out_dir, 'annotations.json'),
                win_width=win_width,
                win_height=win_height,
                hor_overlap=hor_overlap,
                vert_overlap=vert_overlap,
                img_id=img_id,
                obj_id=obj_id,
                save_empty=not remove_empty,
                keep_incomplete=not remove_incomplete,
                task=task,
                progress_callback=pbar.update
            )

    elif ann_type == AnnotationType.PVOC:
        from ginjinn.utils.dataset_cropping import sliding_window_crop_pvoc

        os.mkdir(os.path.join(out_dir, 'annotations'))
        with tqdm.tqdm(total=len(get_image_files(image_dir))) as pbar:
            sliding_window_crop_pvoc(
                img_dir=image_dir,
                ann_dir=ann_path,
                img_dir_out=os.path.join(out_dir, 'images'),
                ann_dir_out=os.path.join(out_dir, 'annotations'),
                win_width=win_width,
                win_height=win_height,
                hor_overlap=hor_overlap,
                vert_overlap=vert_overlap,
                save_empty=not remove_empty,
                keep_incomplete=not remove_incomplete,
                progress_callback=pbar.update
            )

def utils_sw_split(args):
    '''utils_sw_split

    GinJinn utils sw_split command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sw_split subcommand.
    '''

    # process commandline arguments, where necessary
    win_width, win_height = process_window_size(args.window_size)
    hor_overlap, vert_overlap = process_overlap(args.overlap)

    image_dir = args.image_dir
    ann_path = args.ann_path
    dataset_dir = args.dataset_dir
    input_type = process_input(image_dir=image_dir, ann_path=ann_path, dataset_dir=dataset_dir)

    # get remaining commandline arguments
    out_dir = args.out_dir
    task = args.task
    remove_empty = args.remove_empty
    remove_incomplete = args.remove_incomplete
    img_id = args.img_id
    obj_id = args.obj_id

    # check if out_dir exists
    from ginjinn.commandline.commandline_app.commandline_helpers import prepare_out_dir
    prepare_out_dir(out_dir)

    # image splitting
    if input_type == InputType.Images:
        sw_split_images(
            image_dir=image_dir,
            out_dir=out_dir,
            win_width=win_width,
            win_height=win_height,
            hor_overlap=hor_overlap,
            vert_overlap=vert_overlap,
        )

    # dataset splitting
    elif input_type == InputType.Dataset:
        from ginjinn.commandline.commandline_app.commandline_helpers import (
            check_dataset_dir,
            DatasetType,
            AnnotationType,
        )
        ds_type, ann_type, splits = check_dataset_dir(dataset_dir=dataset_dir)

        # is simple dataset
        if ds_type == DatasetType.simple:
            if ann_type == AnnotationType.COCO:
                ann_path = os.path.join(dataset_dir, 'annotations.json')
            elif ann_type == AnnotationType.PVOC:
                ann_path = os.path.join(dataset_dir, 'annotations')
            else:
                print('ERROR: UNEXPECTED ERROR')
                sys.exit(1)

            image_dir = os.path.join(dataset_dir, 'images')

            print(f'Splitting dataset "{dataset_dir}" ...')
            split_images_and_annotations(
                image_dir=image_dir,
                ann_path=ann_path,
                out_dir=out_dir,
                win_width=win_width,
                win_height=win_height,
                hor_overlap=hor_overlap,
                vert_overlap=vert_overlap,
                img_id=img_id,
                obj_id=obj_id,
                remove_empty=remove_empty,
                remove_incomplete=remove_incomplete,
                task=task,
            )
            print(f'Sliding window dataset written to "{out_dir}".')
        # is split dataset
        elif ds_type == DatasetType.split:
            for split_name in splits:
                split_dir = os.path.join(dataset_dir, split_name)
                if ann_type == AnnotationType.COCO:
                    ann_path = os.path.join(split_dir, 'annotations.json')
                elif ann_type == AnnotationType.PVOC:
                    ann_path = os.path.join(split_dir, 'annotations')
                else:
                    print('ERROR: UNEXPECTED ERROR')
                    sys.exit(1)

                image_dir = os.path.join(split_dir, 'images')

                split_out_dir = os.path.join(out_dir, split_name)
                os.mkdir(split_out_dir)

                print(f'Splitting dataset "{split_dir}" ...')
                split_images_and_annotations(
                    image_dir=image_dir,
                    ann_path=ann_path,
                    out_dir=split_out_dir,
                    win_width=win_width,
                    win_height=win_height,
                    hor_overlap=hor_overlap,
                    vert_overlap=vert_overlap,
                    img_id=img_id,
                    obj_id=obj_id,
                    remove_empty=remove_empty,
                    remove_incomplete=remove_incomplete,
                    task=task,
                )
                print(f'Sliding window dataset written to "{split_out_dir}".')

        else:
            print('ERROR: UNEXPECTED ERROR')
            sys.exit(1)

    # annotations + images splitting
    elif input_type == InputType.AnnotationsAndImages:
        print(f'Splitting images from "{image_dir}", and annotations from "{ann_path}" ...')
        split_images_and_annotations(
            image_dir=image_dir,
            ann_path=ann_path,
            out_dir=out_dir,
            win_width=win_width,
            win_height=win_height,
            hor_overlap=hor_overlap,
            vert_overlap=vert_overlap,
            img_id=img_id,
            obj_id=obj_id,
            remove_empty=remove_empty,
            remove_incomplete=remove_incomplete,
            task=task,
        )

    # invalid
    else:
        print('ERROR: input type not supported!')
