''' ginjinn predict parser
'''

import argparse

def setup_predict_parser(subparsers):
    '''setup_predict_parser

    Setup parser for the ginjinn predict subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the predict subcommand.
    '''

    parser = subparsers.add_parser(
        'predict',
        help = '''
            Predict from a trained object detection model.
        ''',
        description = '''
            Predict from a trained object detection model.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    # Required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-i', '--image_path',
        type = str,
        help = '''
            Either path to an image directory or to a single image.
        ''',
        required=True,
    )

    # TODO: refactor this!!!
    # - always return COCO
    # - optionally cropped and vis
    required.add_argument(
        '-s', '--output_types',
        help = '''
            Output types.
        ''',
        choices=['COCO', 'cropped', 'visualization'],
        nargs='+',
        action='append',
        default=['COCO'],
    )

    # Optional
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. By default, output will be written to
            "<project_dir>/prediction".
        ''',
        default = None,
    )

    optional.add_argument(
        '-c', '--checkpoint',
        type = str,
        help = '''
            Checkpoint name. By default model_final.pth will be used.
        ''',
        default = "model_final.pth",
    )

    optional.add_argument(
        '-t', '--threshold',
        type = float,
        help = '''
            Prediction threshold. Only predictions with confidence scores >= threshold are saved.
        ''',
        default = 0.8
    )

    optional.add_argument(
        '-p', '--padding',
        type = int,
        help = '''
            Padding for cropping bounding boxes in pixels.
            Only relevant if cropped output option is enabled.
        ''',
        default = 0
    )

    optional.add_argument(
        '-r', '--seg_refinement',
        dest = 'seg_refinement',
        action = 'store_true',
        help = '''
            Apply segmentation refinement using CascadePSP (https://arxiv.org/abs/2005.02551).
        '''
    )
    parser.set_defaults(seg_refinement = False)

    optional.add_argument(
        '-m', '--refinement_mode',
        help = '''
            Refinement mode. Either "fast" or "full".
        ''',
        choices=['fast', 'full'],
        default='full',
    )

    optional.add_argument(
        '-d', '--device',
        help = '''
            Hardware device to be used for segmentation refinement.
            Since CascadePSP is computationally intensive, it is highly recommended
            to use a GPU device. By default the first available GPU will be used.
        ''',
        type=str,
        default='cuda:0',
    )

    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return parser
