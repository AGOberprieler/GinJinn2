''' ginjinn split parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

def setup_split_parser(subparsers):
    '''setup_split_parser

    Setup parser for the ginjinn split subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the split subcommand.
    '''

    parser = subparsers.add_parser(
        'split',
        help = '''
            Split a single dataset (images and annotations) into train, and optionally
            validation and test datasets.
        ''',
        description = '''
            Split a single dataset (images and annotations) into train, and optionally
            validation and test datasets.

            |R|RThis command can be called by either specifying a dataset directory via -I/--dataset_dir,
            or by specifying an annotation path and an image directory via -i/--image_dir, and 
            -a/--ann_path, respectively.

            |R|RIf -I/--dataset_dir is provided, ginjinn will try to infer the annotations and image
            directores. The dataset directory must be a simple dataset; split datasets are not supported.
            |RExample: ginjinn split -I my_dataset -o my_split_dataset -d instance-segmentation

            |R|R-i/--image_dir and -a/--ann_path can be provided, if the dataset is in a non-standard format.
            |RExample: ginjinn split -a my_annotations.json -i my_images -o my_split_dataset -d instance-segmentation
        ''',
        add_help=False,
        formatter_class=NewlineFormatter,
    )
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. Sub-datasets will be written to <out_dir>/train,
            "<out_dir>/val" or "<out_dir>/test". The output directory
            will be newly created, if it does not exist. 
        ''',
        required = True,
    )

    required.add_argument(
        '-d', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )

    required.add_argument(
        '-I', '--dataset_dir',
        type = str,
        help = '''
            Dataset directory.
        ''',
        required=False,
        default=None,
    )

    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Directory containing annotations (PVOC) or path to an annotation
            JSON file (COCO). Note: this argument may only be used with -i/--image_dir.
        ''',
        required = False,
        default=None,
    )

    required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Image directory. Note: this argument may only be used with -a/--ann_path.
        ''',
        default=None,
        required=False
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-t', '--test_proportion',
        type = float,
        help = '''
            Proportion of the dataset to reserve for final model evaluation (ginjinn evaluate).
        ''',
        default = 0.2,
    )
    optional.add_argument(
        '-v', '--validation_proportion',
        type = float,
        help = '''
            Proportion of the dataset to use for validation while training.
        ''',
        default = 0.2,
    )
    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
