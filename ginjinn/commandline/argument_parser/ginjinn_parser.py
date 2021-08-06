''' ginjinn commandline parser
'''

import argparse

from .evaluate import setup_evaluate_parser
from .info import setup_info_parser
from .new import setup_new_parser
from .predict import setup_predict_parser
from .simulate import setup_simulate_parser
from .split import setup_split_parser
from .train import setup_train_parser
from .utils import setup_utils_parser
from .visualize import setup_visualize_parser

class GinjinnArgumentParser():
    '''GinjinnArgumentParser

    Class for setting up and handling commandline arguments.
    '''

    _description = '''
        GinJinn is a toolbox for simplifying the setup, training, evaluation of object detection models.
        In addition, it provides data pre- and postprocessing features to facilitate the construction of
        custom object detection pipelines.
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=self._description,
            add_help=False
        )
        self.parser.add_argument(
            '-d', '--debug',
            help='Run in debug mode.',
            action='store_true',
        )
        self.parser.add_argument(
            '-h', '--help',
            action='help',
            help='Show this help message and exit.',
        )

        self._subparsers = self.parser.add_subparsers(
            dest='subcommand',
            help='GinJinn subcommands.'
        )
        self._init_subparsers()

    def parse_args(self, args=None, namespace=None):
        '''parse_args
        Parses the commandline arguments and returns them in argparse
        format.

        Parameters
        ----------
        args
            List of strings to parse. If None, the strings are taken from sys.argv.
        namespace
            An object to take the attributes. The default is a new empty argparse Namespace object.

        Returns
        -------
        args
            Parsed argparse arguments
        '''

        return self.parser.parse_args(args=args, namespace=namespace)

    def _init_subparsers(self):
        '''_init_subparsers

        Initilialize parsers for GinJinn subcommands.
        '''

        setup_evaluate_parser(self._subparsers)
        setup_info_parser(self._subparsers)
        setup_new_parser(self._subparsers)
        setup_predict_parser(self._subparsers)
        setup_simulate_parser(self._subparsers)
        setup_split_parser(self._subparsers)
        setup_train_parser(self._subparsers)
        setup_utils_parser(self._subparsers)
        setup_visualize_parser(self._subparsers)
