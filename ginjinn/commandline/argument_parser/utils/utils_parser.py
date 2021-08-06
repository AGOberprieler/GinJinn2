''' ginjinn utils parser
'''

# from .cleanup import setup_cleanup_parser
from .count import setup_count_parser
from .crop import setup_crop_parser
from .filter_cat import setup_filter_cat_parser
from .filter_size import setup_filter_size_parser
from .flatten import setup_flatten_parser
from .flatten_img_dir import setup_flatten_img_dir_parser
from .merge import setup_merge_parser
from .sliding_window import setup_sliding_window_parser
from .sw_image import setup_sw_image_parser
from .sw_split import setup_sw_split_parser
from .sw_merge import setup_sw_merge_parser

def setup_utils_parser(subparsers):
    '''setup_utils_parser

    Setup parser for the ginjinn utils subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the utils subcommand.
    '''

    parser = subparsers.add_parser(
        'utils',
        help = '''
            Auxiliary commands, mainly for data pre- and postprocessing.
        ''',
        description = '''
            Auxiliary commands, mainly for data pre- and postprocessing.
        ''',
    )

    utils_parsers = parser.add_subparsers(
        dest='utils_subcommand',
    )
    utils_parsers.required = True

    # setup subparsers
    # setup_cleanup_parser(utils_parsers)
    setup_count_parser(utils_parsers)
    setup_crop_parser(utils_parsers)
    setup_filter_cat_parser(utils_parsers)
    setup_filter_size_parser(utils_parsers)
    setup_flatten_parser(utils_parsers)
    setup_flatten_img_dir_parser(utils_parsers)
    setup_merge_parser(utils_parsers)
    setup_sliding_window_parser(utils_parsers)
    setup_sw_image_parser(utils_parsers)
    setup_sw_split_parser(utils_parsers)
    setup_sw_merge_parser(utils_parsers)

    return parser
