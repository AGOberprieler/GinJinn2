''' ginjinn utils flatten functionality
'''

import os
from pathlib import Path
import sys
from typing import Tuple, List

import tqdm


def get_image_files_recursive(
    img_dir: str,
    img_file_extensions: Tuple[str] = (
        '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
        '*.png', '*.PNG',
    ),
) -> List[str]:
    '''get_image_files_recursive

    Get paths of image files in img_root_dir recursively.

    Parameters
    ----------
    img_dir : str
        Directory containing images.
    img_file_extensions : List[str], optional
        Image file extensions,
        by default [ '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', ]

    Returns
    -------
    List[str]
        List of image file paths.
    '''
    img_files = []
    for ext in img_file_extensions:
        img_files.extend([str(p) for p in Path(img_dir).rglob(ext)])
    return img_files

def check_image_root_dir(
    image_root_dir : str,
) -> List[str]:
    '''check_image_root_dir

    Check if image_root_dir is a valid image root directory.
    Can exit.

    Parameters
    ----------
    image_root_dir : str
        Image root directory.

    Returns
    -------
    List[str]
        List of image files.
    '''

    if not os.path.exists(image_root_dir):
        print(f'ERROR: "{image_root_dir}" does not exist.')
        sys.exit(1)

    image_files = get_image_files_recursive(image_root_dir)
    if len(image_files) < 1:
        print(
            f'ERROR: "{image_root_dir}" and subdirectories do not '
            'contain any supported image files.'
        )
        sys.exit(1)

    return image_files

def utils_flatten(args):
    '''utils_flatten

    GinJinn utils flatten command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils flatten
        subcommand.
    '''

    out_dir = args.out_dir
    image_root_dir = args.image_root_dir
    ann_path = args.ann_path
    sep = args.separator
    unique_id = args.unique_id
    annotated_only = args.annotated_only

    from ginjinn.commandline.commandline_app.commandline_helpers import (
        prepare_out_dir,
        check_ann_path,
        AnnotationType
    )

    image_files = check_image_root_dir(image_root_dir)

    # images only
    if ann_path is None:
        prepare_out_dir(out_dir)

        from ginjinn.utils.data_prep import flatten_img_dir
        print(f'Flatenning images from "{image_root_dir}" ...')
        with tqdm.tqdm(total = len(image_files)) as pbar:
            flatten_img_dir(
                img_root_dir=image_root_dir,
                out_dir=out_dir,
                sep=sep,
                unique_id=unique_id,
                link_images=True,
                progress_callback=pbar.update,
            )
        print(f'Flattened image directory written to "{out_dir}".')

    # annotations and images
    else:
        ann_type = check_ann_path(ann_path)
        if ann_type != AnnotationType.COCO:
            print(f'ERROR: could not find COCO annotations at "{ann_path}".')
            sys.exit(1)
        prepare_out_dir(out_dir)

        os.mkdir(os.path.join(out_dir, 'images'))
        from ginjinn.utils import flatten_coco
        print(f'Flatenning images from "{image_root_dir}" and annotations from "{ann_path}" ...')
        with tqdm.tqdm(total = len(image_files)) as pbar:
            flatten_coco(
                ann_file=ann_path,
                img_root_dir=image_root_dir,
                out_dir=out_dir,
                sep=sep,
                unique_id=unique_id,
                annotated_only=annotated_only,
                progress_callback=pbar.update
            )
        print(f'Flattened dataset written to "{out_dir}".')
