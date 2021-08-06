''' ginjinn utils filter_cat parser
'''

import argparse

def setup_filter_cat_parser(subparsers):
    '''setup_filter_cat_parser

    Setup parser for the ginjinn utils filter_cat subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils count subcommand.
    '''
    # == filter_cat
    filter_cat_parser = subparsers.add_parser(
        'filter_cat',
        help = '''
            Filter objects by category.
        ''',
        description = '''
            Filter objects by category.
            ATTENTION: By default, only a new annotation file will be created.
            If you are interested in creating a new filtered dataset, provide an image directory
            using -i/--img_dir.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    filter_cat_parser_required = filter_cat_parser.add_argument_group('required arguments')
    filter_cat_parser_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory, to which the filtered annotations, and optionally filtered images
            (see img_dir option) will be written.
        ''',
        required=True,
    )
    filter_cat_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or directory containing PVOC annotations.
        ''',
        required=True,
    )
    filter_cat_parser_required.add_argument(
        '-f', '--filter',
        type = str,
        help = '''
            Names of categories to keep.
            By default, the passed categories are kept and the remaining ones are dropped.
            The -d/--drop option can be used to invert this behavior.
        ''',
        action = 'append',
        required = True,
    )

    # optional
    filter_cat_parser_optional = filter_cat_parser.add_argument_group('optional arguments')
    filter_cat_parser_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. Will be inferred if set to 'auto'.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto',
    )
    filter_cat_parser_optional.add_argument(
        '-d', '--drop',
        action = 'store_true',
        help = '''
            Drop categories in filter instead of keeping them.
        '''
    )
    filter_cat_parser.set_defaults(drop = False)

    filter_cat_parser_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing the annotated images. Use this parameter if you
            want to filter out images without annotation after category filtering.
        ''',
        required=False,
        default=None,
    )
    filter_cat_parser_optional.add_argument(
        '-c', '--copy_images',
        action = 'store_true',
        help = '''
            Copy images to img_dir instead of creating hard links.
        '''
    )
    filter_cat_parser.set_defaults(copy_images = False)

    filter_cat_parser_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return filter_cat_parser
