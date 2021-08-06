''' ginjinn train parser
'''


def setup_train_parser(subparsers):
    '''setup_train_parser

    Setup parser for the ginjinn train subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the train subcommand.
    '''

    parser = subparsers.add_parser(
        'train',
        help = '''
            Train an object detection model.
        ''',
        description = '''
            Train an object detection model.
        ''',
        add_help=False,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            GinJinn project directory.
        '''
    )

    parser.add_argument(
        '-n', '--n_iter',
        type = int,
        help = 'Number of iterations. Overrides the number of iterations given in ginjinn_config.yaml.',
        default = None,
    )

    parser.add_argument(
        '-r',
        '--resume',
        dest='resume',
        action='store_true',
        help='Resume Training. Overrides the resume option given in ginjinn_config.yaml.'
    )
    parser.add_argument(
        '-x',
        '--no-resume',
        dest='resume',
        action='store_false',
        help='''
            Restart training. Overrides the resume option given in ginjinn_config.yaml.
            ATTENTION: This will discard previous training progress!
        '''
    )
    parser.set_defaults(resume=None)

    parser.add_argument(
        '-f', '--force',
        dest='force',
        action='store_true',
        help='Enforce removal of existing outputs when resume is set to False.'
    )
    parser.set_defaults(force=False)

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return parser
