''' Module for the ginjinn info subcommand
'''

import sys

def ginjinn_info(args):
    '''ginjinn_info

    GinJinn info command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn
        info subcommand.
    '''
    from ginjinn.utils.utils import dataset_info, find_img_dir, get_anntype, ImageDirNotFound

    ann_path = args.ann_path
    img_dir = args.img_dir
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

    dataset_info(
        ann_path=ann_path,
        img_dir=img_dir,
        ann_type=ann_type,
    )
