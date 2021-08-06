''' ginjinn utils cleanup parser
'''

import argparse

def setup_cleanup_parser(subparsers):
    '''setup_cleanup_parser

    Setup parser for the ginjinn utils cleanup subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils cleanup subcommand.
    '''

    cleanup_parser = subparsers.add_parser(
        'cleanup',
        help = '''
            Cleanup GinJinn project directory, removing the outputs directory and evaluation an training results.
        ''',
        description = '''
            Cleanup GinJinn project directory, removing the outputs directory and evaluation an training results.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cleanup_parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            GinJinn project directory to be cleaned up.
        ''',
    )

    return cleanup_parser
