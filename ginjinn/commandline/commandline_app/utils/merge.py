''' ginjinn merge functionality
'''

def utils_merge(args):
    '''utils_merge

    GinJinn utils merge command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils merge
        subcommand.
    '''

    from ginjinn.data_reader.merge_datasets import merge_datasets_coco, merge_datasets_pvoc

    image_dirs = [x[0] for x in args.image_dir]
    ann_paths = [x[0] for x in args.ann_path]

    out_dir = args.out_dir
    ann_type = args.ann_type

    link_images = args.link_images

    if ann_type == 'COCO':
        merge_datasets_coco(
            ann_files=ann_paths,
            img_dirs=image_dirs,
            outdir=out_dir,
            link_images=link_images,
        )
    elif ann_type == 'PVOC':
        merge_datasets_pvoc(
            ann_dirs=ann_paths,
            img_dirs=image_dirs,
            outdir=out_dir,
            link_images=link_images,
        )