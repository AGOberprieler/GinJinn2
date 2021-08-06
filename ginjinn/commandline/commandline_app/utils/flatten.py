''' ginjinn utils flatten functionality
'''

import os
import sys
import shutil

def utils_flatten(args):
    '''utils_flatten

    GinJinn utils flatten command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils flatten
        subcommand.
    '''

    from ginjinn.utils import confirmation_cancel
    from ginjinn.utils import flatten_coco

    out_dir = args.out_dir
    image_root_dir = args.image_root_dir
    ann_path = args.ann_path
    sep = args.separator
    unique_id = args.unique_id
    annotated_only = args.annotated_only

    if os.path.exists(out_dir):
        if confirmation_cancel(
            f'\nDirectory "{out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{out_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(out_dir)
        else:
            sys.exit()

    os.mkdir(out_dir)

    flatten_coco(
        ann_file=ann_path,
        img_root_dir=image_root_dir,
        out_dir=out_dir,
        sep=sep,
        unique_id=unique_id,
        annotated_only=annotated_only,
    )

    print(f'Flattened dataset written to {out_dir}.')
