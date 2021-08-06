''' ginjinn utils count functionality
'''

import os

def utils_count(args):
    '''utils_count

    GinJinn utils count command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        count subcommand.
    '''

    from ginjinn.utils.utils import count_categories, count_categories_pvoc

    if os.path.isdir(args.ann_path):
        count_df = count_categories_pvoc(args.ann_path)
    else:
        count_df = count_categories(args.ann_path)
    count_df.to_csv(args.out_file, index_label='image')

    print(f'\nCategory counts written to {args.out_file}.')
