''' ginjinn utils sw_merge parser
'''

import argparse

def setup_sw_merge_parser(subparsers):
    '''setup_sw_merge_parser

    Setup parser for the ginjinn utils sw_merge subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils sw_merge subcommand.
    '''
    # == sw_merge
    sw_merge_parser = subparsers.add_parser(
        'sw_merge',
        help = '''
            Merge sliding-window cropped images and annotations.
            Objects will be merged only if they satisfy the intersection threshold and,
            additionally, at least one of the IoU and IoS thresholds.
        ''',
        description = '''
            Merge sliding-window cropped images and annotations.
            Objects will be merged only if they satisfy the intersection threshold and,
            additionally, at least one of the IoU and IoS thresholds.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    sw_merge_required = sw_merge_parser.add_argument_group('required arguments')
    sw_merge_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. Will be created if it does not exist.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Path to directory containing the sliding-window cropped images.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-t', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )

    # optional
    sw_merge_optional = sw_merge_parser.add_argument_group('optional arguments')
    sw_merge_optional.add_argument(
        '-c', '--intersection_threshold',
        type = int,
        help = '''
            Absolute intersection threshold for merging in pixel.
        ''',
        default=0,
    )
    sw_merge_optional.add_argument(
        '-u', '--iou_threshold',
        type = float,
        help = '''
            Intersection over Union (IoU) threshold for merging in pixel.
            Only calculated within the window overlap.
        ''',
        default=0.5,
    )
    sw_merge_optional.add_argument(
        '-s', '--ios_threshold',
        type = float,
        help = '''
            Intersection over Smaller Object (IoS) threshold for merging in pixel.
        ''',
        default=0.8,
    )
    sw_merge_optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    # == other utils
    # ...

    return sw_merge_parser
