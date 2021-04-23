''' Module for the simulate subcommand.
'''

import os
import sys
import shutil

from ginjinn.utils import confirmation_cancel
from ginjinn.simulation import generate_simple_shapes_coco, generate_simple_shapes_pvoc


def ginjinn_simulate(args):
    '''ginjinn_simulate

    GinJinn simulate command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn simulate
        subcommand.

    Raises
    ------
    Exception
        Raised if unknown subcommand passed.
    '''
    if args.simulate_subcommand == 'shapes':
        simulate_shapes(args)
    else:
        err = f'Unknown simulate subcommand "{args.simulate_subcommand}".'
        raise Exception(err)

def simulate_shapes(args):
    '''simulate_shapes

    GinJinn simulate shapes command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn simulate shapes
        subcommand.
    '''

    out_dir = args.out_dir

    if os.path.exists(out_dir):
        if confirmation_cancel(
            f'\nDirectory "{out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{out_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(out_dir)
        else:
            sys.exit()

    if args.ann_type == 'COCO':
        img_dir = os.path.join(out_dir, 'images')
        ann_path = os.path.join(out_dir, 'annotations.json')

        os.mkdir(out_dir)
        os.mkdir(img_dir)

        circle_col = args.circle_col.lstrip('#')
        triangle_col = args.triangle_col.lstrip('#')
        circle_col =  [int(circle_col[i:i+2], 16) / 255 for i in (0, 2, 4)]
        triangle_col =  [int(triangle_col[i:i+2], 16) / 255 for i in (0, 2, 4)]

        generate_simple_shapes_coco(
            img_dir=img_dir,
            ann_file=ann_path,
            n_images=args.n_images,
            min_w=args.min_w,
            max_w=args.max_w,
            min_h=args.min_h,
            max_h=args.max_h,
            min_n_shapes=args.min_n_shapes,
            max_n_shapes=args.max_n_shapes,
            circle_col=circle_col,
            triangle_col=triangle_col,
            col_var=args.color_variance,
            min_r=args.min_shape_radius,
            max_r=args.max_shape_radius,
            min_rot=args.min_shape_angle,
            max_rot=args.max_shape_angle,
            noise=args.noise,
        )

        print(f'Simulated dataset written to "{out_dir}".')

    if args.ann_type == 'PVOC':
        img_dir = os.path.join(out_dir, 'images')
        ann_dir = os.path.join(out_dir, 'annotations')

        os.mkdir(out_dir)
        os.mkdir(img_dir)
        os.mkdir(ann_dir)

        circle_col = args.circle_col.lstrip('#')
        triangle_col = args.triangle_col.lstrip('#')
        circle_col =  [int(circle_col[i:i+2], 16) / 255 for i in (0, 2, 4)]
        triangle_col =  [int(triangle_col[i:i+2], 16) / 255 for i in (0, 2, 4)]

        generate_simple_shapes_pvoc(
            img_dir=img_dir,
            ann_dir=ann_dir,
            n_images=args.n_images,
            min_w=args.min_w,
            max_w=args.max_w,
            min_h=args.min_h,
            max_h=args.max_h,
            min_n_shapes=args.min_n_shapes,
            max_n_shapes=args.max_n_shapes,
            circle_col=circle_col,
            triangle_col=triangle_col,
            col_var=args.color_variance,
            min_r=args.min_shape_radius,
            max_r=args.max_shape_radius,
            min_rot=args.min_shape_angle,
            max_rot=args.max_shape_angle,
            noise=args.noise,
        )

        print(f'Simulated dataset written to "{out_dir}".')
