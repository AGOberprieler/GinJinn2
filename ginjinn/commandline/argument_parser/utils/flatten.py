''' ginjinn utils flatten parser
'''

from ginjinn.commandline.argument_parser.argparse_utils import NewlineFormatter

def setup_flatten_parser(subparsers):
    '''setup_flatten_parser

    Setup parser for the ginjinn utils flatten subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils flatten subcommand.
    '''
    # == flatten
    flatten_parser = subparsers.add_parser(
        'flatten',
        help = '''
            Flatten image directory, or COCO dataset: move all images into a single, unnested directory.
            In case of COCO input, annotations are adjusted accordingly.
        ''',
        description = '''
            Flatten image directory, or COCO dataset: move all images into a single, unnested directory.
            In case of COCO input, annotations are adjusted accordingly.
            |R|RFlatten image directories by providing the -i/--image_root_dir argument only:
            |Rginjinn utils flatten -i my_nested_images -o my_flattened_images

            |R|RFlatten a COCO dataset by providing -i/--image_root_dir and -a/--ann_path:
            |Rginjinn utils flatten -i my_dataset/images -a my_dataset/annotations.json -o my_flattened_dataset
        ''',
        formatter_class=NewlineFormatter,
        add_help=False,
    )

    # required
    flatten_required = flatten_parser.add_argument_group('required arguments')

    flatten_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the flattened dataset will be written to.
        ''',
        required=True,
    )
    flatten_required.add_argument(
        '-i', '--image_root_dir',
        type = str,
        help = '''
            Root image directory. For COCO this is generally the "images" directory
            within the COCO dataset directory.
        ''',
        required=True,
    )
    flatten_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to a COCO (JSON) annotation file.
        ''',
        required=False,
    )

    # optional
    flatten_optional = flatten_parser.add_argument_group('optional arguments')
    flatten_optional.add_argument(
        '-s', '--separator',
        type = str,
        help = '''
            The new image file names will encode the original image paths.
            The separator will replace the "/" characters in the latter.
            For example, the image file "image_root_dir/a/b/c.jpg" may result in
            "a~b~c.jpg" (unless -u/--unique_id is specified).
        ''',
        default='~',
    )
    flatten_optional.add_argument(
        '-u', '--unique_id',
        dest = 'unique_id',
        action = 'store_true',
        help = '''
            Replace image file names with an unique id. An ID mapping file
            will be written if this option is set.
        '''
    )
    flatten_parser.set_defaults(unique_id = False)

    flatten_optional.add_argument(
        '-x', '--annotated_only',
        dest = 'annotated_only',
        action = 'store_true',
        help = '''
            Keep only images with corresponding object annotations.
            Only relevant for COCO input.
        '''
    )
    flatten_parser.set_defaults(annotated_only = False)

    flatten_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return flatten_parser
