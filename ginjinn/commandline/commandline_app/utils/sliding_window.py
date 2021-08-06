''' ginjinn utils sliding_window functionality
'''

import os
import sys
import shutil

def utils_sliding_window(args):
    '''utils_sliding_window

    GinJinn utils sliding_window command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sliding_window subcommand.
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

    from ginjinn.utils.dataset_cropping import sliding_window_crop_coco, sliding_window_crop_pvoc
    from ginjinn.utils.utils import get_anntype, find_img_dir, ImageDirNotFound
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

    img_dir_out = os.path.join(args.out_dir, 'images')
    if os.path.exists(img_dir_out):
        msg = f'Directory "{img_dir_out} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{img_dir_out}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(img_dir_out)
            os.mkdir(img_dir_out)
    else:
        os.mkdir(img_dir_out)

    ann_path = args.ann_path
    ann_type = args.ann_type
    img_dir = args.img_dir

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

    if ann_type == 'COCO':
        ann_path_out = os.path.join(args.out_dir, 'annotations.json')

        sliding_window_crop_coco(
            img_dir=img_dir,
            ann_path=ann_path,
            img_dir_out=img_dir_out,
            ann_path_out=ann_path_out,
            win_width=win_width,
            win_height=win_height,
            hor_overlap=hor_overlap,
            vert_overlap=vert_overlap,
            img_id=args.img_id,
            obj_id=args.obj_id,
            save_empty=not args.remove_empty,
            keep_incomplete=not args.remove_incomplete,
            task=args.task,
        )

        msg = f'Sliding-window cropped images written to {img_dir_out}. '+\
            f'Sliding-window cropped annotation written to {ann_path_out}.'
        print(msg)

    elif ann_type == 'PVOC':
        ann_dir_out = os.path.join(args.out_dir, 'annotations')
        if os.path.exists(ann_dir_out):
            msg = f'Directory "{ann_dir_out} already exists. Should it be overwritten?"\n' +\
                f'WARNING: This will remove "{ann_dir_out}" and ALL SUBDIRECTORIES.\n'
            should_remove = confirmation_cancel(msg)
            if should_remove:
                shutil.rmtree(ann_dir_out)
                os.mkdir(ann_dir_out)
        else:
            os.mkdir(ann_dir_out)

        sliding_window_crop_pvoc(
            img_dir=img_dir,
            ann_dir=ann_path,
            img_dir_out=img_dir_out,
            ann_dir_out=ann_dir_out,
            win_width=win_width,
            win_height=win_height,
            hor_overlap=hor_overlap,
            vert_overlap=vert_overlap,
            save_empty=not args.remove_empty,
            keep_incomplete=not args.remove_incomplete,
        )

        msg = f'Sliding-window cropped images written to {img_dir_out}. '+\
            f'Sliding-window cropped annotations written to {ann_dir_out}.'
        print(msg)
