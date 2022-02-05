''' ginjinn utils crop parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter


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
    parser = subparsers.add_parser(
        'crop',
        help='''
            Crop bounding boxes found in a COCO dataset as single images.
            This can be useful for multi-step pipelines, e.g. training a bounding box model
            based on the original images, and a segmentation model using the cropped bounding
            boxes.
        ''',
        description='''
            Crop bounding boxes found in a COCO dataset as single images.
            This can be useful for multi-step pipelines, e.g. training a bounding box model
            based on the original images, and a segmentation model using the cropped bounding
            boxes.

            |RInput can be specified by providing one of either -I/--dataset_dir, or -a/--ann_path AND
            -i/--image_dir.
            |R|RIf -I/--dataset_dir is provided, this command will expect a simple or split COCO dataset.
            |RExample: ginjinn utils crop -I my_dataset -o my_dataset_cropped -t bbox

            |R|RIf -a/--ann_path and -i/--image_dir are provided, objects in the COCO annotations file will be cropped
            from the images in image_dir.
            |RExample: ginjinn utils crop -a my_annotations.json -i my_images -o my_dataset_cropped -t bbox
        ''',
        formatter_class=NewlineFormatter,
        add_help=False,
    )

    # required
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '-o',
        '--out_dir',
        type=str,
        help='''
            Output directory.
        ''',
        required=True,
    )
    required.add_argument(
        '-t',
        '--cropping_type',
        type=str,
        help='''
            Cropping type. When "segmentation" is selected,
            only bounding boxes with a corresponding segmentation will be cropped.
        ''',
        choices=['segmentation', 'bbox'],
        required=True,
    )
    required.add_argument(
        '-I',
        '--dataset_dir',
        type=str,
        help='''
            Simple or split COCO dataset directory.
        ''',
        required=False,
    )
    required.add_argument(
        '-a',
        '--ann_path',
        type=str,
        help='''
            COCO (JSON) annotation file. Can only be used with -i/--image_dir.
        ''',
        required=False,
    )
    required.add_argument(
        '-i',
        '--image_dir',
        type=str,
        help='''
            Image directory. Can only be used with -a/--ann_path.
        ''',
        required=False,
    )

    # optional
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-p',
        '--padding',
        type=int,
        help='''
            Padding for cropping bounding boxes in pixels.
        ''',
        default=5,
    )
    optional.add_argument(
        '-h', '--help', action='help', help='Show this help message and exit.'
    )

    return parser
