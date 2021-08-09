''' Module for the ginjinn utils subcommand.
'''

from .cleanup import utils_cleanup
from .count import utils_count
from .crop import utils_crop
from .filter_cat import utils_filter_cat
from .filter_size import utils_filter_size
from .flatten import utils_flatten
from .merge import utils_merge
from .sw_merge import utils_sw_merge
from .sw_split import utils_sw_split

def ginjinn_utils(args):
    '''ginjinn_utils

    GinJinn utils command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        subcommand.

    Raises
    ------
    Exception
        Raised if an unknown utils subcommand is passed.
    '''

    if args.utils_subcommand == 'merge':
        utils_merge(args)
    elif args.utils_subcommand == 'cleanup':
        utils_cleanup(args)
    elif args.utils_subcommand == 'flatten':
        utils_flatten(args)
    elif args.utils_subcommand == 'crop':
        utils_crop(args)
    elif args.utils_subcommand == 'sw_split':
        utils_sw_split(args)
    elif args.utils_subcommand == 'sw_merge':
        utils_sw_merge(args)
    elif args.utils_subcommand == 'filter_cat':
        utils_filter_cat(args)
    elif args.utils_subcommand == 'filter_size':
        utils_filter_size(args)
    elif args.utils_subcommand == 'count':
        utils_count(args)
    else:
        err = f'Unknown utils subcommand "{args.utils_subcommand}".'
        raise Exception(err)
