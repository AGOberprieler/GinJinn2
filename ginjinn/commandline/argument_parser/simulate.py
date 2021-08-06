''' ginjinn simulate parser
'''

import argparse

# TODO: modularize this

def setup_simulate_parser(subparsers):
    '''setup_simulate_parser

    Setup parser for the ginjinn simulate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the simulate subcommand.
    '''

    parser = subparsers.add_parser(
        'simulate',
        help = '''
            Create simulated datasets.
        ''',
        description = '''
            Create simulated datasets.
        ''',
    )
    simulate_parsers = parser.add_subparsers(
        dest='simulate_subcommand',
        help='Types of simulations.'
    )

    # == shapes
    shapes_parser = simulate_parsers.add_parser(
        'shapes',
        help = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        description = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    required = shapes_parser.add_argument_group('required arguments')
    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the simulated data should be written to.
        ''',
        required=True,
    )

    optional = shapes_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-a', '--ann_type',
        type = str,
        help = '''
            Type of annotations to simulate.
        ''',
        choices=['COCO', 'PVOC'],
        default='COCO',
    )
    optional.add_argument(
        '-n', '--n_images',
        type = int,
        help = '''
            Number of images to simulate.
        ''',
        default=100,
    )
    optional.add_argument(
        '-w0', '--min_w',
        type = int,
        help = '''
            Minimum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-w1', '--max_w',
        type = int,
        help = '''
            Maximum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h0', '--min_h',
        type = int,
        help = '''
            Minimum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h1', '--max_h',
        type = int,
        help = '''
            Maximum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-n0', '--min_n_shapes',
        type = int,
        help = '''
            Minimum number of shapes per image.
        ''',
        default=1,
    )
    optional.add_argument(
        '-n1', '--max_n_shapes',
        type = int,
        help = '''
            Maximum number of shapes per image.
        ''',
        default=4,
    )
    optional.add_argument(
        '-t', '--triangle_prob',
        type = float,
        help = '''
            Probability of generating a triangle. Default is 0.5, meaning that
            triangles and circle are equally represented.
        ''',
        default=0.5,
    )
    optional.add_argument(
        '-ccol', '--circle_col',
        type = str,
        help = '''
            Mean circle color as Hex color code.
        ''',
        default='#C87D7D',
    )
    optional.add_argument(
        '-tcol', '--triangle_col',
        type = str,
        help = '''
            Mean triangle color as Hex color code.
        ''',
        default='#7DC87D',
    )
    optional.add_argument(
        '-cvar', '--color_variance',
        type = float,
        help = '''
            Variance around mean shape colors.
        ''',
        default=0.15,
    )
    optional.add_argument(
        '-mnr', '--min_shape_radius',
        type = float,
        help = '''
            Minimum shape radius.
        ''',
        default=25.0,
    )
    optional.add_argument(
        '-mxr', '--max_shape_radius',
        type = float,
        help = '''
            Maximum shape radius.
        ''',
        default=75.0,
    )
    optional.add_argument(
        '-mna', '--min_shape_angle',
        type = float,
        help = '''
            Minimum shape rotation in degrees.
        ''',
        default=0.0,
    )
    optional.add_argument(
        '-mxa', '--max_shape_angle',
        type = float,
        help = '''
            Maximum shape rotation in degrees.
        ''',
        default=60.0,
    )
    optional.add_argument(
        '-b', '--noise',
        type = float,
        help = '''
            Amount of noise to add.
        ''',
        default=0.005,
    )
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    # ==
    # ... further simulations ...
    # ==

    return parser
