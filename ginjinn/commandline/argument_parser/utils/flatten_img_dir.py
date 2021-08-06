''' ginjinn utils flatten_img_dir parser
'''

import argparse

def setup_flatten_img_dir_parser(subparsers):
    '''setup_flatten_img_dir_parser

    Setup parser for the ginjinn utils flatten_img_dir subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils flatten_img_dir subcommand.
    '''
    # == flatten_img_dir
    flatten_img_dir_parser = subparsers.add_parser(
        'flatten_img_dir',
        help = '''
            Flatten image directory: move all images into a single, unnested directory.
        ''',
        description = '''
            Flatten image directory: move all images into a single, unnested directory.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    # required
    flatten_img_dir_required = flatten_img_dir_parser.add_argument_group('required arguments')

    flatten_img_dir_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the images should be linked to.
        ''',
        required=True,
    )

    flatten_img_dir_required.add_argument(
        '-i', '--image_root_dir',
        type = str,
        help = '''
            Path to image directory root.
        ''',
        required=True,
    )

    # optional
    flatten_img_dir_optional = flatten_img_dir_parser.add_argument_group('optional arguments')
    flatten_img_dir_optional.add_argument(
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
    flatten_img_dir_optional.add_argument(
        '-u', '--unique_id',
        dest = 'unique_id',
        action = 'store_true',
        help = '''
            Replace image file names with a unique id. An ID mapping file
            will be written if this option is set.
        '''
    )
    flatten_img_dir_parser.set_defaults(unique_id = False)

    flatten_img_dir_optional.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit.'
    )

    return flatten_img_dir_parser