''' visualize command
'''

import os
import shutil
import sys

from ginjinn.utils import confirmation_cancel

def ginjinn_visualize(args):
    '''visualize

    GinJinn visualize command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn visualize.
    '''

    from ginjinn.utils.utils import visualize_annotations

    if os.path.exists(args.out_dir):
        msg = f'Directory "{args.out_dir} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{args.out_dir}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
    else:
        os.mkdir(args.out_dir)

    from ginjinn.utils.utils import find_img_dir, ImageDirNotFound, get_anntype

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
                'explicitly pass "--img_dir".'
            )
            sys.exit()

    if ann_type == 'auto':
        ann_type = get_anntype(ann_path)

    visualize_annotations(
        ann_path = ann_path,
        img_dir = img_dir,
        out_dir = args.out_dir,
        ann_type = ann_type,
        vis_type = args.vis_type,
    )

    print(f'Visualizations written to "{args.out_dir}".')
