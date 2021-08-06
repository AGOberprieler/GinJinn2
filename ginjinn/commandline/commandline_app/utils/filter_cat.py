''' ginjinn utils filter_cat functionality
'''

def utils_filter_cat(args):
    '''utils_filter_cat

    GinJinn utils filter_cat command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        filter_cat subcommand.
    '''

    from ginjinn.utils.data_prep import filter_categories_coco, filter_categories_pvoc
    from ginjinn.utils.utils import get_anntype

    ann_path = args.ann_path
    ann_type = args.ann_type
    if ann_type == 'auto':
        ann_type = get_anntype(ann_path)

    if ann_type == 'COCO':
        filter_categories_coco(
            ann_file = ann_path,
            img_dir = args.img_dir,
            out_dir = args.out_dir,
            drop = args.filter if args.drop else None,
            keep = args.filter if not args.drop else None,
            link_images = not args.copy_images,
        )
    elif ann_type == 'PVOC':
        filter_categories_pvoc(
            ann_dir = ann_path,
            img_dir = args.img_dir,
            out_dir = args.out_dir,
            drop = args.filter if args.drop else None,
            keep = args.filter if not args.drop else None,
            link_images = not args.copy_images,
        )
    else:
        print(f'Unknown annotation type "{args.ann_type}".')
        return

    print(f'Filtered annotations written to "{args.out_dir}".')
