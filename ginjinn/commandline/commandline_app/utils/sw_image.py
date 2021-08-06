''' ginjinn utils sw_image functionality
'''

import os
import sys
import shutil

def utils_sw_image(args):
    '''utils_sw_image

    GinJinn utils sw_image command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sw_image subcommand.
    '''
    window_size = args.window_size
    if len(window_size) == 1:
        win_width = window_size[0]
        win_height = window_size[0]
    elif len(window_size) == 2:
        win_width = window_size[0]
        win_height = window_size[1]
    else:
        print('ERROR: "-s/--window_size" must receive 1 or 2 arguments.')
        return

    overlap = args.overlap
    if len(overlap) == 1:
        hor_overlap = overlap[0]
        vert_overlap = overlap[0]
    elif len(overlap) == 2:
        hor_overlap = overlap[0]
        vert_overlap = overlap[1]
    else:
        print('ERROR: "-p/--overlap" must receive 1 or 2 arguments.')
        return

    from ginjinn.utils.dataset_cropping import sliding_window_crop_images
    from ginjinn.utils import confirmation_cancel

    if os.path.exists(args.out_dir):
        msg = f'Directory "{args.out_dir} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{args.out_dir}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
    else:
        os.mkdir(args.out_dir)

    img_dir = args.img_dir

    if not os.path.isdir(img_dir):
        msg = f'ERROR: "{img_dir}" is not a directory.'
        print(msg)
        sys.exit()

    sliding_window_crop_images(
        img_dir=img_dir,
        img_dir_out=args.out_dir,
        win_width=win_width,
        win_height=win_height,
        hor_overlap=hor_overlap,
        vert_overlap=vert_overlap,
    )

    msg = f'Sliding-window cropped images written to {args.out_dir}.'
    print(msg)
