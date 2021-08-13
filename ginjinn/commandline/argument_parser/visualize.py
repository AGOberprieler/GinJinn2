''' ginjinn visualize parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

def setup_visualize_parser(subparsers):
    '''setup_visualize_parser

    Setup parser for the ginjinn visualize subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the visualize subcommand.
    '''

    parser = subparsers.add_parser(
        'visualize',
        help = '''
            Visualize object annotations on images.
        ''',
        description = '''
            Visualize object annotations on images.

            |RInput can be specified using either -I/--dataset_dir, or -a/--ann_path and -i/--image_dir.

            |R|RIf -I/--dataset_dir is provided, a simple dataset (COCO or PVOC) is expected. In this case,
            -o/--out_dir is not required and defaults to "out_dir/visualization"; it can however still be provided
            to change the visualization output directory.
            |RExamples:
            |R  ginjinn visualize -I my_dataset -v segmentation
            |R  ginjinn visualize -I my_dataset -o my_visualization -v segmentation

            |R|RIf -i/--image_dir and -a/--ann_path are provided, the images and the corresponding annotations
            will visualized. Here, -o/--out_dir is required.
            |RExample:
            |R  ginjinn visualize -a my_annotations.json -i my_images -o my_visualization -v bbox
        ''',
        aliases=['vis'],
        add_help=False,
        formatter_class=NewlineFormatter,
    )

    # required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-v', '--vis_type',
        type = str,
        help = '''
            Visualization type. Either "bbox" for bounding-boxes or "segmentation"
            for segmentation masks. For PVOC, only "bbox" is allowed.
        ''',
        choices = ['segmentation', 'bbox'],
        required=True,
    )
    required.add_argument(
        '-I', '--dataset_dir',
        type = str,
        help = '''
            Dataset directory.
        ''',
        required=False,
    )
    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or PVOC annotation directory.
            Can only be used with -i/--image_dir AND -o/--out_dir.
        ''',
        required=False,
    )
    required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Directory containing (potentially a subset of) the annotated images.
            Can only be used with -a/--ann_path AND -o/--out_dir.
        ''',
        default=None,
        required=False,
    )
    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory.
            If -I/--dataset_dir is provided, visualizations will be written to
            "out_dir/visualization" by default.
            If -i/--image_dir and -a/--ann_path are provided, this argument is required.
        ''',
        required=False,
    )

    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return parser
