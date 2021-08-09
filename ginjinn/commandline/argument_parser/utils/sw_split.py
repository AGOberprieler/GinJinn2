''' ginjinn utils sw_split parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

def setup_sw_split_parser(subparsers):
    '''setup_sw_split_parser

    Setup parser for the ginjinn utils sw_split subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils sw_split subcommand.
    '''
    parser = subparsers.add_parser(
        'sw_split',
        help = '''
            Split images, simple datasets, or split datasets into sliding windows.
        ''',
        description = '''
            Split images, simple datasets, or split datasets into sliding windows.
            |R|RThe general behavior of this command is determined by the three input parameters
            -i/--image_dir, -a/--ann_path, and -I/--input_dir:

            |R|RIf -I/--dataset_dir is provided, a simple dataset (COCO or PVOC), or a split dataset as generated
            by ginjinn split will be subjected to sliding-window splitting.
            |RExample: ginjinn sw_split -I my_dataset -o my_split_dataset

            |R|RIf only -i/--image_dir is provided, images will be split into sliding windows.
            |RExample: ginjinn sw_split -i my_images -o my_split_images

            |R|RIf -i/--image_dir and -a/--ann_path are provided, the images and the corresponding annotations
            will be split into sliding windows.
            |RExample: ginjinn sw_split -a my_dataset/annotations.json -i my_dataset/images -o my_split_dataset
        ''',
        formatter_class=NewlineFormatter,
        add_help=False,
    )

    # required
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '-I', '--dataset_dir',
        type = str,
        help = '''
            Dataset or split dataset directory.
        ''',
        required=False,
    )

    required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Image directory.
        ''',
        required=False,
    )

    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or directory containing PVOC annotations.
            Note: this argument may only be used with -i/--image_dir.
        ''',
        required=False,
    )

    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. Will be created if it does not exist.
        ''',
        required=True,
    )

    # optional
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-s', '--window_size',
        type = int,
        nargs = '+',
        help = '''
            Sliding window size in pixel.
            If one argument is passed, quadratic windows of window_size will be generated.
            If two arguments are passed, they are interpreted as window width and height, respectively.

            "-s 500", for example, crops sliding windows of size 500*500 (w*h), while "-s 500 300" crops 
            windows of size 500*300.
        ''',
        default=[1024],
    )
    optional.add_argument(
        '-p', '--overlap',
        type = int,
        nargs = '+',
        help = '''
            Overlap between sliding windows in pixel.
            If one argument is passed, the same overlap is used in horizontal and vertical direction.
            If two arguments are passed, they are interpreted as overlap in horizontal and
            vertical direction, respectively.
        ''',
        default=[256],
    )
    optional.add_argument(
        '-k', '--task',
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        default = 'instance-segmentation',
    )
    optional.add_argument(
        '-r', '--remove_empty',
        dest = 'remove_empty',
        action = 'store_true',
        help = '''
            If this flag is set, cropped images without object annotation will
            not be saved.
        '''
    )
    parser.set_defaults(remove_empty = False)
    optional.add_argument(
        '-c', '--remove_incomplete',
        dest = 'remove_incomplete',
        action = 'store_true',
        help = '''
            If this flag is set, object annotations that are touched (trimmed)
            by a sliding-window edge are removed from the corresponding sliding-window images. 
        '''
    )
    parser.set_defaults(remove_incomplete = False)
    optional.add_argument(
        '-m', '--img_id',
        type = int,
        help = '''
            Starting image ID for newly generated image annotations. Relevant for COCO annotations only.
        ''',
        default=1,
    )
    optional.add_argument(
        '-b', '--obj_id',
        type = int,
        help = '''
            Starting object ID for newly generated object annotations. Relevant for COCO annotations only.
        ''',
        default=1,
    )
    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
