''' ginjinn utils merge
'''

import argparse

def setup_merge_parser(subparsers):
    '''setup_merge_parser

    Setup parser for the ginjinn utils merge subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils merge subcommand.
    '''
    # == merge
    merge_parser = subparsers.add_parser(
        'merge',
        help = '''
            Merge multiple datasets into a single one.
        ''',
        description = '''
            Merge multiple datasets into a single one.
            All input datasets must be of the same annotation type (COCO or PVOC).
            To merge two COCO datasets A and B into C, the merge command
            might be called like this:
            ginjinn utils merge -t COCO -i A/images -i B/images -a A/annotations.json -a B/annotations.json -o C
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    merge_parser_required = merge_parser.add_argument_group('required arguments')

    merge_parser_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Ouput directory, which the merged dataset will be written to.
        ''',
        required=True,
    )

    merge_parser_required.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type of the datasets.
        ''',
        choices=['COCO', 'PVOC'],
    )

    merge_parser_required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            A single image directory.
            This option must be provided for each input dataset.
            See description.
        ''',
        required=True,
        nargs='+',
        action='append',
    )

    merge_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            A single annotation file for COCO (JSON) or a single annotation directory for PVOC.
            This option must be provided for each input dataset.
            See description.
        ''',
        required=True,
        nargs='+',
        action='append',
    )

    # optional
    merge_parser_optional = merge_parser.add_argument_group('optional arguments')

    merge_parser_optional.add_argument(
        '-l', '--link_images',
        dest = 'link_images',
        action = 'store_true',
        help = '''
            Create hard links instead of copying images.
        '''
    )
    merge_parser.set_defaults(link_images = False)

    merge_parser_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return merge_parser