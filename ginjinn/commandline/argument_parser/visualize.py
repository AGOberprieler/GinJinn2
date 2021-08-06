''' ginjinn visualize parser
'''


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

    visualize_parser = subparsers.add_parser(
        'visualize',
        help = '''
            Visualize object annotations on images.
        ''',
        description = '''
            Visualize object annotations on images.
        ''',
        aliases=['vis'],
        add_help=False,
    )

    # required
    visualize_parser_required = visualize_parser.add_argument_group('required arguments')
    visualize_parser_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Directory the visualizations should be written to.
        ''',
        required=True,
    )
    visualize_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or PVOC annotation directory.
        ''',
        required=True,
    )
    visualize_parser_required.add_argument(
        '-v', '--vis_type',
        type = str,
        help = '''
            Visualization type. Either "bbox" for bounding-boxes or "segmentation"
            for segmentation masks. For PVOC, only "bbox" is allowed.
        ''',
        choices = ['segmentation', 'bbox'],
        required=True,
    )

    visualize_parser_optional = visualize_parser.add_argument_group('optional arguments')
    visualize_parser_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing (potentially a subset of) the annotated images.
            By default, will be inferred.
        ''',
        default = None,
    )
    visualize_parser_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If "auto", will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto'
    )

    visualize_parser_optional.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    return visualize_parser
