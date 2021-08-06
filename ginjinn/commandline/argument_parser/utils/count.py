''' ginjinn utils count parser
'''

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
    count_parser = subparsers.add_parser(
        'count',
        help = '''
            Count objects per category for each image.
        ''',
        description = '''
            Count objects per category for each image.
        ''',
        add_help=False,
    )

    # required
    count_parser_required = count_parser.add_argument_group('required arguments')
    count_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON).
        ''',
        required=True,
    )
    count_parser_required.add_argument(
        '-o', '--out_file',
        type = str,
        help = '''
            File (CSV) the category counts should be written to.
        ''',
        required=True,
    )

    count_parser_optional = count_parser.add_argument_group('optional arguments')
    count_parser_optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return count_parser
