''' ginjinn split parser
'''


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
        ''',
        add_help=False,
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-a', '--annotation_path',
        type = str,
        help = '''
            Path to directory containing annotations (PVOC) or path to an annotation
            JSON file (COCO).
        ''',
        required = True,
    )
    required.add_argument(
        '-o', '--output_dir',
        type = str,
        help = '''
            Output directory. Sub-datasets will be written to OUTPUT_DIR/train,
            OUTPUT_DIR/val or OUTPUT_DIR/test. The output directory
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
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing the annotated images.
            By default, images are expected to be found in a sister directory of ANN_PATH.
        ''',
        default=None,
    )
    optional.add_argument(
        '-k', '--ann_type',
        type = str,
        choices = ['auto', 'COCO', 'PVOC'],
        help = '''
            Annotation type. Will be inferred if set to 'auto'.
        ''',
        default='auto',
    )
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
    optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return parser
