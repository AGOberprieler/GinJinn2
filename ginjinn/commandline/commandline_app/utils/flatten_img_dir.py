''' ginjinn utils flatten_img_dir functionality
'''

import os
import sys
import shutil

def utils_flatten_img_dir(args):
    '''utils_flatten_img_dir

    GinJinn utils flatten_img_dir command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        flatten_img_dir subcommand.
    '''

    from ginjinn.utils.data_prep import flatten_img_dir
    from ginjinn.utils import confirmation_cancel

    out_dir = args.out_dir
    image_root_dir = args.image_root_dir
    sep = args.separator
    unique_id = args.unique_id

    if os.path.exists(out_dir):
        if confirmation_cancel(
            f'\nDirectory "{out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{out_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(out_dir)
        else:
            sys.exit()

    os.mkdir(out_dir)

    flatten_img_dir(
        img_root_dir=image_root_dir,
        out_dir=out_dir,
        sep=sep,
        unique_id=unique_id,
        link_images=True,
    )

    print(f'Flattened images written to {out_dir}.')
