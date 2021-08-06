''' ginjinn utils sliding_window parser
'''

import argparse

def setup_sliding_window_parser(subparsers):
    '''setup_sliding_window_parser

    Setup parser for the ginjinn utils sliding_window subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils sliding_window subcommand.
    '''

    sliding_window_parser = subparsers.add_parser(
        'sliding_window',
        help = '''
            Crop images and corresponding annotation into sliding windows.
        ''',
        description = '''
            Crop images and corresponding annotation into sliding windows.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    sliding_window_required = sliding_window_parser.add_argument_group('required arguments')

    sliding_window_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the sliding-window cropped dataset should be written to.
        ''',
        required=True,
    )
    sliding_window_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file for COCO annotations or
            path to a directory containing XML annotations for PVOC annotations.
        ''',
        required=True,
    )
    sliding_window_required.add_argument(
        '-k', '--task',
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for. Only applies to COCO
            datasets.
        ''',
    )

    # optional
    sliding_window_optional = sliding_window_parser.add_argument_group('optional arguments')
    sliding_window_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Path to image directory. By default, will be inferred.
        ''',
        default = None,
    )
    sliding_window_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If "auto", will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto',
    )
    sliding_window_optional.add_argument(
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
    sliding_window_optional.add_argument(
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
    sliding_window_optional.add_argument(
        '-r', '--remove_empty',
        dest = 'remove_empty',
        action = 'store_true',
        help = '''
            If this flag is set, cropped images without object annotation will
            not be saved.
        '''
    )
    sliding_window_parser.set_defaults(remove_empty = False)
    sliding_window_optional.add_argument(
        '-c', '--remove_incomplete',
        dest = 'remove_incomplete',
        action = 'store_true',
        help = '''
            If this flag is set, object annotations that are touched (trimmed)
            by a sliding-window edge are removed from the corresponding sliding-window images. 
        '''
    )
    sliding_window_parser.set_defaults(remove_incomplete = False)
    sliding_window_optional.add_argument(
        '-m', '--img_id',
        type = int,
        help = '''
            Starting image ID for newly generated image annotations.
        ''',
        default=1,
    )
    sliding_window_optional.add_argument(
        '-b', '--obj_id',
        type = int,
        help = '''
            Starting object ID for newly generated object annotations.
        ''',
        default=1,
    )

    sliding_window_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return sliding_window_parser
