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

    # Positional
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
        '-c', '--crop',
        help = '''
            Crop predicted bounding boxes and segmentation masks (segmentation models only) from
            input images. Bounding boxes and masks are written to "<out_dir>/images_cropped" and
            "<out_dir>/masks_cropped", respectively. In case of instance segmentation, an
            additional COCO json file with annotations referring to the cropped images (bounding boxes)
            will be written to "<out_dir>/annotations_cropped.json".
        ''',
        dest='crop',
        action = 'store_true',
    )
    parser.set_defaults(crop = False)

    optional.add_argument(
        '-v', '--visualize',
        help = '''
            Visualize predictions on input images. Visualization output is
            written to "<out_dir>/visualization"
        ''',
        dest='visualize',
        action = 'store_true',
    )
    parser.set_defaults(visualize = False)

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
            Only relevant if crop output option is enabled.
        ''',
        default = 0
    )

    optional.add_argument(
        '-w', '--weights_checkpoint',
        type = str,
        help = '''
            Checkpoint name. By default model_final.pth will be used.
        ''',
        default = "model_final.pth",
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

    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
