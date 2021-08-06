''' info parser
'''

def setup_info_parser(subparsers):
    '''setup_info_parser

    Setup parser for the ginjinn info subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the info subcommand.
    '''

    parser = subparsers.add_parser(
        'info',
        help = '''
            Summarize basic information about a dataset (number of images, objects per category).
        ''',
        description = '''
            Summarize basic information about a dataset (number of images, objects per category).
        ''',
        add_help=False,
    )

    # required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or directory containing PVOC annotations.
        ''',
        required=True,
    )

    # optional
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing the annotated images. By default, images are expected to be found in a sister directory
            of ANN_PATH.
        ''',
        default = None,
    )
    optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. Will be inferred if set to 'auto'.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto'
    )
    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
