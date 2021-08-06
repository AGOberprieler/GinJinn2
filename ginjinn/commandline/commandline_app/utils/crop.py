''' ginjinn utils crop functionality
'''

import sys

def utils_crop(args):
    '''utils_crop

    GinJinn utils crop command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils crop
        subcommand.
    '''

    from ginjinn.utils import crop_seg_from_coco
    from ginjinn.utils import crop_bbox_from_coco
    from ginjinn.utils.utils import find_img_dir, ImageDirNotFound

    img_dir = args.img_dir
    ann_path = args.ann_path

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

    if args.type == 'segmentation':
        crop_seg_from_coco(
            ann_file=ann_path,
            img_dir=img_dir,
            outdir=args.out_dir,
            padding=args.padding,
        )
    else:
        crop_bbox_from_coco(
            ann_file=ann_path,
            img_dir=img_dir,
            outdir=args.out_dir,
            padding=args.padding,
        )

    print(
        f'Cropped dataset written to "{args.out_dir}".'
    )
