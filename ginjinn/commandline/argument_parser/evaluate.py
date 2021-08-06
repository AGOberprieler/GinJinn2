''' ginjinn evaluate parser
'''

import argparse


def setup_evaluate_parser(subparsers: argparse.ArgumentParser):
    '''setup_evaluate_parser

    Setup parser for the ginjinn evaluate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the evaluate subcommand.
    '''

    parser = subparsers.add_parser(
        'evaluate',
        aliases=['eval'],
        help='''
            Evaluate a trained object detection model.
            This calculates COCO evaluation metrics (see https://cocodataset.org/#detection-eval) for the test dataset and stores the results
            in a CSV file.
        ''',
        description='''
            Evaluate a trained object detection model.
            This calculates COCO evaluation metrics (see https://cocodataset.org/#detection-eval) for the test dataset and stores the results
            in a CSV file.
            Example usage:

            ginjinn evaluate my_analysis
        ''',
        add_help=False,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-c', '--checkpoint',
        type = str,
        help = '''
            Checkpoint name. By default model_final.pth will be used.
        ''',
        default = "model_final.pth",
    )
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return parser
