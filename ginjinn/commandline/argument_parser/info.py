''' info parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

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

            |R|RThis command can be called by either specifying a dataset directory via -I/--dataset_dir,
            or by specifying an annotation path and an image directory via -i/--image_dir, and 
            -a/--ann_path, respectively.

            |R|RIf -I/--dataset_dir is provided, ginjinn will try to infer the annotations and image
            directores. The dataset directory must be a simple dataset; split datasets are not supported.
            |RExample: ginjinn info -I my_dataset

            |R|R-i/--image_dir and -a/--ann_path can be provided, if the dataset is in a non-standard format.
            |RExample: ginjinn info -a my_annotations.json -i my_images
        ''',
        formatter_class=NewlineFormatter,
        add_help=False,
    )

    # required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-I', '--dataset_dir',
        type = str,
        help = '''
            Dataset directory.
        ''',
        required=False,
    )

    required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Image directory. Note: this argument may only be used with -a/--ann_path.
        ''',
        required=False,
    )

    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or directory containing PVOC annotations.
            Note: this argument may only be used with -i/--image_dir.
        ''',
        required=False,
    )

    # optional
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
