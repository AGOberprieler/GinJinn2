''' ginjinn utils filter_size parser
'''

import argparse

def setup_filter_size_parser(subparsers):
    '''setup_filter_size_parser

    Setup parser for the ginjinn utils filter_size subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils filter_size subcommand.
    '''
    # == filter_size
    filter_size_parser = subparsers.add_parser(
        'filter_size',
        help = '''
            Filter objects by size.
            Note: This function is only compatible with annotations in COCO format.
        ''',
        description = '''
            Filter objects by size.
            Note: This function is only compatible with annotations in COCO format.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    filter_size_parser_required = filter_size_parser.add_argument_group('required arguments')
    filter_size_parser_required.add_argument(
        '-o', '--out_file',
        type = str,
        help = '''
            COCO annotation file (JSON), which the filtered annotations should be written to.
        ''',
        required=True,
    )
    filter_size_parser_required.add_argument(
        '-a', '--ann_file',
        type = str,
        help = '''
            Path to COCO annotation file (JSON).
        ''',
        required=True,
    )
    filter_size_parser_required.add_argument(
        '-d', '--task',
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
    filter_size_parser_optional = filter_size_parser.add_argument_group(
        'optional arguments'
    )
    filter_size_parser_optional.add_argument(
        '-x', '--min_width',
        type = int,
        default = 5,
        help = '''
            Minimal total object width in pixels.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-y', '--min_height',
        type = int,
        default = 5,
        help = '''
            Minimal total object height in pixels.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-r', '--min_area',
        type = int,
        default = 25,
        help = '''
            Minimal total object area in pixels.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-f', '--min_fragment_area',
        type = int,
        default = 25,
        help = '''
            Minimal object fragment area in pixels.
            Fragments of non-contiguous objects smaller than this will be discarded.
        '''
    )

    filter_size_parser_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return filter_size_parser
