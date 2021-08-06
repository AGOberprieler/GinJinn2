''' ginjinn utils sw_image parser
'''

import argparse

def setup_sw_image_parser(subparsers):
    '''setup_sw_image_parser

    Setup parser for the ginjinn utils sw_image subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils sw_image subcommand.
    '''
    sw_image_parser = subparsers.add_parser(
        'sw_image',
        help = '''
            Crop images into sliding windows.
        ''',
        description = '''
            Crop images into sliding windows.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    sw_image_required = sw_image_parser.add_argument_group('required arguments')

    sw_image_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the sliding-window images should be written to.
        ''',
        required=True,
    )
    sw_image_required.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Path to image directory. By default, will be inferred.
        ''',
        required = True,
    )

    # optional
    sw_image_optional = sw_image_parser.add_argument_group('optional arguments')
    sw_image_optional.add_argument(
        '-s', '--window_size',
        type = int,
        nargs = '+',
        help = '''
            Sliding window size in pixel.
            If one argument is passed, quadratic windows of window_size will be generated.
            If two arguments are passed, they are interpreted as window width and height, respectively.

            "-s 500", for example, crops sliding windows of size 500*500 (w*h), while "-s 500 300" crops 
            windows of size 500*300.
        ''',
        default=[1000],
    )
    sw_image_optional.add_argument(
        '-p', '--overlap',
        type = int,
        nargs = '+',
        help = '''
            Overlap between sliding windows.
            If one argument is passed, the same overlap is used in horizontal and vertical direction.
            If two arguments are passed, they are interpreted as overlap in horizontal and
            vertical, respectively.
        ''',
        default=[200],
    )

    return sw_image_parser
