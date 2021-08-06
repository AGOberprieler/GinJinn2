''' ginjinn utils sw_split functionality
'''

import os
import sys
import shutil

def utils_sw_split(args):
    '''utils_sw_split

    GinJinn utils utils_sw_split command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sw_split subcommand.
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
    from ginjinn.utils import confirmation_cancel

    if os.path.exists(args.out_dir):
        msg = f'Directory "{args.out_dir} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{args.out_dir}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
        else:
            print('sw_split canceled')
            return
    else:
        os.mkdir(args.out_dir)


    from ginjinn.utils.utils import get_dstype

    ann_type = args.ann_type
    # infer ann_type
    if ann_type == 'auto':
        ds_types = []
        for ds_name in ['train', 'val', 'test']:
            ds_path = os.path.join(args.split_dir, ds_name)
            if not os.path.isdir(ds_path):
                continue
            ds_types.append(get_dstype(ds_path))
        if len(set(ds_types)) > 1:
            print(
                f'ERROR: Found multiple dataset types in "{args.split_dir}". ' +\
                'The datasets in splitdir must all have the same annotation type (COCO or PVOC).'
            )
            sys.exit()
        if len(ds_types) < 1:
            print(
                f'ERROR: Could not find any dataset (train, val, test) in "{args.split_dir}".'
            )
            sys.exit()
        ann_type = ds_types[0]

    if ann_type == 'COCO':
        for ds_name in ['train', 'val', 'test']:
            img_dir = os.path.join(args.split_dir, ds_name, 'images')
            ann_path = os.path.join(args.split_dir, ds_name, 'annotations.json')

            if not os.path.isdir(img_dir):
                print(
                    f'No image directory found for dataset "{ds_name}". ' +\
                    f'(Expected location: "{img_dir}")'
                )
                continue
            if not os.path.isfile(ann_path):
                print(
                    f'No annotation file found for dataset "{ds_name}". ' +\
                    f'(Expected location: "{ann_path}")'
                )
                continue

            img_dir_out = os.path.join(args.out_dir, ds_name, 'images')
            os.makedirs(img_dir_out, exist_ok=True)
            ann_path_out = os.path.join(args.out_dir, ds_name, 'annotations.json')

            print(f'Splitting dataset {ds_name}...')
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

            msg = \
                f'Sliding-window images for dataset {ds_name} written to {img_dir_out}. '+\
                f'Sliding-window annotation for dataset {ds_name} written to {ann_path_out}.'
            print(msg)

    elif ann_type == 'PVOC':
        for ds_name in ['train', 'val', 'test']:
            img_dir = os.path.join(args.split_dir, ds_name, 'images')
            ann_dir = os.path.join(args.split_dir, ds_name, 'annotations')

            if not os.path.isdir(img_dir):
                print(
                    f'No image directory found for dataset "{ds_name}". ' +\
                    f'(Expected location: "{img_dir}")'
                )
                continue
            if not os.path.isdir(ann_dir):
                print(
                    f'No annotation directory found for dataset "{ds_name}". ' +\
                    f'(Expected location: "{ann_dir}")'
                )
                continue

            img_dir_out = os.path.join(args.out_dir, ds_name, 'images')
            os.makedirs(img_dir_out, exist_ok=True)
            ann_dir_out = os.path.join(args.out_dir, ds_name, 'annotations')
            os.makedirs(ann_dir_out, exist_ok=True)

            print(f'Splitting dataset {ds_name}...')
            sliding_window_crop_pvoc(
                img_dir=img_dir,
                ann_dir=ann_dir,
                img_dir_out=img_dir_out,
                ann_dir_out=ann_dir_out,
                win_width=win_width,
                win_height=win_height,
                hor_overlap=hor_overlap,
                vert_overlap=vert_overlap,
                save_empty=not args.remove_empty,
                keep_incomplete=not args.remove_incomplete,
            )

            msg = \
                f'Sliding-window images for dataset {ds_name} written to {img_dir_out}. '+\
                f'Sliding-window annotations for dataset {ds_name} written to {ann_dir_out}.'
            print(msg)
