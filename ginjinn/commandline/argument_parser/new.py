''' ginjinn new parser
'''

import argparse
from os.path import join, basename

import pkg_resources
import glob

def setup_new_parser(subparsers):
    '''setup_new_parser

    Setup parser for the ginjinn new subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the new subcommand.
    '''

    parser = subparsers.add_parser(
        'new',
        help = '''Create a new GinJinn project.''',
        description = '''Create a new GinJinn project.''',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''GinJinn project directory to be created.'''
    )

    required = parser.add_argument_group('required arguments')

    template_dir =  pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/templates',
    )
    template_files = glob.glob(join(template_dir, '*.yaml'))
    templates = sorted([basename(t_f) for t_f in template_files])
    templates = [t for t in templates if not t.startswith('adv_')]
    templates_string = '\n'.join(f'- {t}' for t in templates)
    required.add_argument(
        '-t', '--template',
        type = str,
        help = f'''Model template, specifying the Detectron2 model to use.
Faster RCNN models are used for bounding-box detection, while
Mask RCNN models are used for instance segmentation. Please do not
exchange the model after project initialization.

Available templates are:
{templates_string}

(default: "faster_rcnn_R_50_FPN_3x.yaml")''',
        choices=templates,
        # default='faster_rcnn_R_50_FPN_3x.yaml',
        required=True,
        metavar='TEMPLATE'
    )

    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '-d', '--data_dir',
        type=str,
        default=None,
        help='''Data directory to initialize the project config for. Can either be the path
to a single COCO/PVOC dataset directory, or a directory comprising multiple datasets
as generated by "ginjinn split".'''
    )

    optional.add_argument(
        '-a', '--advanced',
        dest='advanced',
        action='store_true',
        help='Expose advanced options in the GinJinn configuration file.'
    )
    parser.set_defaults(advanced=False)

    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return parser
