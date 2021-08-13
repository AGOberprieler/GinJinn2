''' ginjinn utils count parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

def setup_count_parser(subparsers):
    '''setup_count_parser

    Setup parser for the ginjinn utils count subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils count subcommand.
    '''
    parser = subparsers.add_parser(
        'count',
        help = '''
            Count objects per category for each image.
        ''',
        description = '''
            Count objects per category for each image.

            |RInput can be specified using either one of -I/--dataset_dir and -a/--ann_path.
            |R|RIf -I/--dataset_dir is provided, this command will expect a simple dataset (COCO or PVOC).
            |RExample: ginjinn utils count -I my_dataset -o count.csv

            |R|RIf -a/--ann_path is provided, objects in the annotations file (COCO) or annotations
            directory (PVOC) will be counted.
            |RExample: ginjinn utils count -a my_annotations.json -o counts.csv
        ''',
        add_help=False,
        formatter_class=NewlineFormatter,
    )

    # required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-o', '--out_file',
        type = str,
        help = '''
            File (CSV) the category counts should be written to.
        ''',
        required=True,
    )

    required.add_argument(
        '-I', '--dataset_dir',
        type = str,
        help = '''
            Dataset directory.
        ''',
        required=False,
    )

    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or PVOC annotations directory.
        ''',
        required=False,
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-s', '--show',
        help='Output counts to terminal. A CSV file will still be generated.',
        action='store_true',
    )

    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
