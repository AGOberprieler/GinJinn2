''' ginjinn utils sw_merge functionality
'''

def utils_sw_merge(args):
    '''utils_sw_merge

    GinJinn utils sw_merge command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sw_merge subcommand.
    '''

    from ginjinn.utils.sliding_window_merging import merge_sliding_window_predictions
    from ginjinn.utils import confirmation_cancel

    def on_out_dir_exists(out_dir):
        return confirmation_cancel(
            f'\nDirectory "{out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{out_dir}" and ALL SUBDIRECTORIES!\n'
        )

    def on_img_out_dir_exists(img_out_dir):
        return confirmation_cancel(
            f'\nDirectory "{img_out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{img_out_dir}" and ALL SUBDIRECTORIES!\n'
        )

    merge_sliding_window_predictions(
        img_dir=args.image_dir,
        ann_path=args.ann_path,
        out_dir=args.out_dir,
        task=args.task,
        iou_threshold=args.iou_threshold,
        ios_threshold=args.ios_threshold,
        intersection_threshold=args.intersection_threshold,
        on_out_dir_exists=on_out_dir_exists,
        on_img_out_dir_exists=on_img_out_dir_exists,
    )

    msg = f'Merging results written to {args.out_dir}.'
    print(msg)
