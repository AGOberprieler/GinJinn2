''' ginjinn utils filter_size functionality
'''

def utils_filter_size(args):
    '''utils_filter_size

    GinJinn utils filter_size command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        filter_size subcommand.
    '''

    from ginjinn.utils.data_prep import filter_objects_by_size

    filter_objects_by_size(
        ann_file = args.ann_file,
        out_file = args.out_file,
        task = args.task,
        min_width = args.min_width,
        min_height = args.min_height,
        min_area = args.min_area,
        min_fragment_area = args.min_fragment_area,
    )

    print(f'Filtered annotation written to "{args.out_file}".')
