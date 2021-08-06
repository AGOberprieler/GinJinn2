''' ginjinn utils crop parser
'''

import argparse

def setup_crop_parser(subparsers):
    '''setup_crop_parser

    Setup parser for the ginjinn utils crop subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils crop subcommand.
    '''
    crop_parser = subparsers.add_parser(
        'crop',
        help = '''
            Crop bounding boxes found in a COCO dataset as single images.
            This can be useful for multi-step pipelines, e.g. training a bounding box model
            based on the original images, and a segmentation model using the cropped bounding
            boxes.
        ''',
        description = '''
            Crop bounding boxes found in a COCO dataset as single images.
            This can be useful for multi-step pipelines, e.g. training a bounding box model
            based on the original images, and a segmentation model using the cropped bounding
            boxes.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    crop_required = crop_parser.add_argument_group('required arguments')

    crop_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory.
        ''',
        required=True,
    )
    crop_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            COCO (JSON) annotation file.
        ''',
        required=True,
    )
    crop_required.add_argument(
        '-t', '--type',
        type=str,
        help='''
            Cropping type. When "segmentation" is selected,
            only bounding boxes with a corresponding segmentation will be cropped.
        ''',
        choices=['segmentation', 'bbox'],
        required=True,
    )

    # optional
    crop_optional = crop_parser.add_argument_group('optional arguments')
    crop_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing the annotated images. By default, images are expected to be found in a sister directory
            of ANN_PATH.
        ''',
        default=None,
    )
    crop_optional.add_argument(
        '-p', '--padding',
        type = int,
        help = '''
            Padding for cropping bounding boxes in pixels.
        ''',
        default=5,
    )
    crop_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return crop_parser
