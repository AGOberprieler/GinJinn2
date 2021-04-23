''' Module for setting up and handling argument parsers
'''

import argparse
import glob
from os.path import basename, join
import pkg_resources

def _setup_new_parser(subparsers):
    '''_setup_new_parser

    Setup parser for the ginjinn new subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the new subcommand.
    '''

    parser = subparsers.add_parser(
        'new',
        help = '''
            Create a new GinJinn project.
        ''',
        description = '''
            Create a new GinJinn project.
        ''',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to new GinJinn project directory.
        '''
    )

    template_dir =  pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/templates',
    )
    template_files = glob.glob(join(template_dir, '*.yaml'))
    templates = sorted([basename(t_f) for t_f in template_files])
    templates = [t for t in templates if not t.startswith('adv_')]
    templates_string = '\n'.join(f'- {t}' for t in templates)
    parser.add_argument(
        '-t', '--template',
        type = str,
        help = f'''
Model template to initialize the new GinJinn project with.
Faster RCNN models are bounding-box detection models, while
Mask RCNN models are segmentation models.

Available templates are:
{templates_string}

(default: "faster_rcnn_R_50_FPN_3x.yaml")
        ''',
        choices=templates,
        default='faster_rcnn_R_50_FPN_3x.yaml',
        metavar='TEMPLATE'
    )

    parser.add_argument(
        '-a', '--advanced',
        dest='advanced',
        action='store_true',
        help='Generate config exposing advanced options.'
    )
    parser.set_defaults(advanced=False)

    parser.add_argument(
        '-d', '--data_dir',
        type=str,
        default=None,
        help='''
            Data directory to initialize the project config for. Can either be
            either the path to a COCO/PVOC dataset directory, or the path to a split
            directory as built by "ginjinn split".
        '''
    )

    return parser

def _setup_train_parser(subparsers):
    '''_setup_train_parser

    Setup parser for the ginjinn train subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the train subcommand.
    '''

    parser = subparsers.add_parser(
        'train',
        help = '''
            Train a GinJinn model.
        ''',
        description = '''
            Train a GinJinn model.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )
    # parser.add_argument(
    #     '-nr', '--no_resume',
    #     type = bool,
    #     help = '''
    #         Do not resume training. If this option is set, training will
    #         start from scratch, discarding previous training checkpoints
    #         PERMANENTLY.
    #     ''',
    #     # action='store_true',
    #     default=None,
    # )

    parser.add_argument(
        '-n', '--n_iter',
        type = int,
        help = 'Number of iterations.',
        default = None,
    )

    parser.add_argument('-r', '--resume', dest='resume', action='store_true')
    parser.add_argument('-nr', '--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=None)

    parser.add_argument(
        '-f', '--force',
        dest='force',
        action='store_true',
        help='Force removal of existing outputs when resume is set to False.'
    )
    parser.set_defaults(force=False)

    return parser

def _setup_evaluate_parser(subparsers):
    '''_setup_evaluate_parser

    Setup parser for the ginjinn evaluate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the evaluate subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'evaluate',
        aliases=['eval'],
        help = '''
            Evaluate a trained GinJinn model.
        ''',
        description = '''
            Evaluate a trained GinJinn model.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-c', '--checkpoint',
        type = str,
        help = '''
            Checkpoint file name. By default the most recent checkpoint
            (model_final.pth) will be used.
        ''',
        default = "model_final.pth",
    )

    return parser

def _setup_predict_parser(subparsers):
    '''_setup_predict_parser

    Setup parser for the ginjinn predict subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the predict subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'predict',
        help = '''
            Predict from a trained GinJinn model.
        ''',
        description = '''
            Predict from a trained GinJinn model.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    # Required
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-i', '--image_path',
        type = str,
        help = '''
            Either path to an image directory or to a single image.
        ''',
        required=True,
    )

    # Optional
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. If None, output will be written to
            "<project_dir>/prediction".
        ''',
        default = None,
    )

    optional.add_argument(
        '-c', '--checkpoint',
        type = str,
        help = '''
            Checkpoint file name. By default the most recent checkpoint
            (model_final.pth) will be used.
        ''',
        default = "model_final.pth",
    )

    optional.add_argument(
        '-t', '--threshold',
        type = float,
        help = '''
            Prediction threshold. Only predictions with scores >= threshold are saved.
        ''',
        default = 0.8
    )

    optional.add_argument(
        '-p', '--padding',
        type = int,
        help = '''
            Padding for cropping bounding boxes.
        ''',
        default = 0
    )

    required.add_argument(
        '-s', '--output_types',
        help = '''
            Output types.
        ''',
        choices=['COCO', 'cropped', 'visualization'],
        nargs='+',
        action='append',
        default=['COCO'],
    )

    optional.add_argument(
        '-r', '--seg_refinement',
        dest = 'seg_refinement',
        action = 'store_true',
        help = '''
            <EXPERIMENTAL> Apply segmentation refinement.
        '''
    )
    parser.set_defaults(seg_refinement = False)

    optional.add_argument(
        '-m', '--refinement_method',
        help = '''
            Refinement method. Either "fast" or "full".
        ''',
        choices=['fast', 'full'],
        default='full',
    )

    optional.add_argument(
        '-d', '--device',
        help = '''
            Refinement device.
        ''',
        type=str,
        default='cuda:0',
    )

    return parser

def _setup_split_parser(subparsers):
    '''_setup_split_parser

    Setup parser for the ginjinn split subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the split subcommand.
    '''

    parser = subparsers.add_parser(
        'split',
        help = '''
            Split dataset (images and annotations) into test, train, and optionally
            evaluation datasets.
        ''',
        description = '''
            Split dataset (images and annotations) into test, train, and optionally
            evaluation datasets.
        '''
    )
    required_parser = parser.add_argument_group('required named arguments')
    required_parser.add_argument(
        '-a', '--annotation_path',
        type = str,
        help = '''
            Path to directory containing annotations (PVOC) or path to an annotation
            JSON file (COCO).
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-o', '--output_dir',
        type = str,
        help = '''
            Path to output directory. Splits will be written to output_dir/train,
            output_dir/test, and output_dir/eval, respectively. The output directory
            will be created, if it does not exist. 
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-d', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )
    optional_parser = parser.add_argument_group('optional arguments')
    optional_parser.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Path to directory containing images. By default, GinJinn searches for
            a sibling directory to "annotation_path" called "images".
        ''',
        default=None,
    )
    optional_parser.add_argument(
        '-k', '--ann_type',
        type = str,
        choices = ['auto', 'COCO', 'PVOC'],
        help = '''
            Annotation type. If 'auto', annotation type will be inferred.
        ''',
        default='auto',
    )
    optional_parser.add_argument(
        '-t', '--test_fraction',
        type = float,
        help = '''
            Fraction of the dataset to use for testing. (Default: 0.2)
        ''',
        default = 0.2,
    )
    optional_parser.add_argument(
        '-v', '--validation_fraction',
        type = float,
        help = '''
            Fraction of the dataset to use for validation while training. (Default: 0.2)
        ''',
        default = 0.2,
    )

    return parser

def _setup_info_parser(subparsers):
    '''_setup_info_parser

    Setup parser for the ginjinn info subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the info subcommand.
    '''

    info_parser = subparsers.add_parser(
        'info',
        help = '''
            Print dataset info.
        ''',
        description = '''
            Print dataset info.
        ''',
    )

    # required
    info_parser_required = info_parser.add_argument_group('required arguments')
    info_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or PVCO annotation directory.
        ''',
        required=True,
    )
    info_parser_optional = info_parser.add_argument_group('optional arguments')
    info_parser_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing  the annotated images.
        ''',
        default = None,
    )
    info_parser_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If 'auto', annotation type will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto'
    )

    return info_parser

def _setup_simulate_parser(subparsers):
    '''_setup_simulate_parser

    Setup parser for the ginjinn simulate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the simulate subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'simulate',
        help = '''
            Simulate datasets.
        ''',
        description = '''
            Simulate datasets.
        ''',
    )
    simulate_parsers = parser.add_subparsers(
        dest='simulate_subcommand',
        help='Types of simulations.'
    )

    # == shapes
    shapes_parser = simulate_parsers.add_parser(
        'shapes',
        help = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        description = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = shapes_parser.add_argument_group('required arguments')
    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the simulated data should be written to.
        ''',
        required=True,
    )

    optional = shapes_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-a', '--ann_type',
        type = str,
        help = '''
            Type of annotations to simulate.
        ''',
        choices=['COCO', 'PVOC'],
        default='COCO',
    )
    optional.add_argument(
        '-n', '--n_images',
        type = int,
        help = '''
            Number of images to simulate.
        ''',
        default=100,
    )
    optional.add_argument(
        '-w0', '--min_w',
        type = int,
        help = '''
            Minimum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-w1', '--max_w',
        type = int,
        help = '''
            Maximum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h0', '--min_h',
        type = int,
        help = '''
            Minimum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h1', '--max_h',
        type = int,
        help = '''
            Maximum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-n0', '--min_n_shapes',
        type = int,
        help = '''
            Minimum number of shapes per image.
        ''',
        default=1,
    )
    optional.add_argument(
        '-n1', '--max_n_shapes',
        type = int,
        help = '''
            Maximum number of shapes per image.
        ''',
        default=4,
    )
    optional.add_argument(
        '-t', '--triangle_prob',
        type = float,
        help = '''
            Probability of generating a triangle. Default is 0.5, meaning that
            triangles and circle are equally represented.
        ''',
        default=0.5,
    )
    optional.add_argument(
        '-ccol', '--circle_col',
        type = str,
        help = '''
            Mean circle color as Hex color code.
        ''',
        default='#C87D7D',
    )
    optional.add_argument(
        '-tcol', '--triangle_col',
        type = str,
        help = '''
            Mean triangle color as Hex color code.
        ''',
        default='#7DC87D',
    )
    optional.add_argument(
        '-cvar', '--color_variance',
        type = float,
        help = '''
            Variance around mean shape colors.
        ''',
        default=0.15,
    )
    optional.add_argument(
        '-mnr', '--min_shape_radius',
        type = float,
        help = '''
            Minimum shape radius.
        ''',
        default=25.0,
    )
    optional.add_argument(
        '-mxr', '--max_shape_radius',
        type = float,
        help = '''
            Maximum shape radius.
        ''',
        default=75.0,
    )
    optional.add_argument(
        '-mna', '--min_shape_angle',
        type = float,
        help = '''
            Minimum shape rotation in degrees.
        ''',
        default=0.0,
    )
    optional.add_argument(
        '-mxa', '--max_shape_angle',
        type = float,
        help = '''
            Maximum shape rotation in degrees.
        ''',
        default=60.0,
    )
    optional.add_argument(
        '-b', '--noise',
        type = float,
        help = '''
            Amount of noise to add.
        ''',
        default=0.005,
    )

    # ==
    # ... further simulations ...
    # ==

    return parser

def _setup_utils_parser(subparsers):
    '''_setup_utils_parser

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
            Utility commands.
        ''',
        description = '''
            Utility commands.
        ''',
    )

    utils_parsers = parser.add_subparsers(
        dest='utils_subcommand',
        help='Utility commands.',
    )
    utils_parsers.required = True

    # == cleanup
    cleanup_parser = utils_parsers.add_parser(
        'cleanup',
        help = '''
            Cleanup GinJinn project directory, removing the outputs directory and evaluation an training results.
        ''',
        description = '''
            Cleanup GinJinn project directory, removing the outputs directory and evaluation an training results.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cleanup_parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            GinJinn project directory to be cleaned up.
        ''',
    )

    # == merge
    merge_parser = utils_parsers.add_parser(
        'merge',
        help = '''
            Merge multiple data sets.
        ''',
        description = '''
            Merge multiple data sets.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    required = merge_parser.add_argument_group('required arguments')

    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the merged data set should be written to.
        ''',
        required=True,
    )

    required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Path to a single image directory.
        ''',
        required=True,
        nargs='+',
        action='append',
    )

    required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to a single annotation file (COCO) or annotations directory (PVOC).
        ''',
        required=True,
        nargs='+',
        action='append',
    )

    # optional
    optional = merge_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type of the data set.
        ''',
        choices=['COCO', 'PVOC'],
        default='COCO',
    )

    optional.add_argument(
        '-l', '--link_images',
        dest = 'link_images',
        action = 'store_true',
        help = '''
            Create hard links instead of copying images.
        '''
    )
    parser.set_defaults(link_images = False)

    # == flatten
    flatten_parser = utils_parsers.add_parser(
        'flatten',
        help = '''
            Flatten a COCO data set (move all images in same directory).
        ''',
        description = '''
            Flatten a COCO data set (move all images in same directory).
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    flatten_required = flatten_parser.add_argument_group('required arguments')

    flatten_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the flattened data set should be written to.
        ''',
        required=True,
    )

    flatten_required.add_argument(
        '-i', '--image_root_dir',
        type = str,
        help = '''
            Path to root image directory. For COCO this is generally the "images" directory
            within the COCO data set directory.
        ''',
        required=True,
    )

    flatten_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file.
        ''',
        required=True,
    )

    # optional
    flatten_optional = flatten_parser.add_argument_group('optional arguments')
    flatten_optional.add_argument(
        '-s', '--seperator',
        type = str,
        help = '''
            Seperator for the image path flattening.
        ''',
        default='~',
    )
    flatten_optional.add_argument(
        '-c', '--custom_id',
        dest = 'custom_id',
        action = 'store_true',
        help = '''
            Replace image file names with a custom id. An ID mapping file
            will be written if this option is set.
        '''
    )
    parser.set_defaults(custom_id = False)

    flatten_optional.add_argument(
        '-x', '--annotated_only',
        dest = 'annotated_only',
        action = 'store_true',
        help = '''
            Whether only annotated images should be kept in the data set.
        '''
    )
    parser.set_defaults(annotated_only = False)

    # == crop
    crop_parser = utils_parsers.add_parser(
        'crop',
        help = '''
            Crop COCO dataset bounding boxes as single images.
            This is useful for multi-step models, e.g. training a bbox model
            and a segmentation model on the cropped bboxes.
        ''',
        description = '''
            Crop COCO dataset bounding boxes as single images.
            This is useful for multi-step models, e.g. training a bbox model
            and a segmentation model on the cropped bboxes.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    crop_required = crop_parser.add_argument_group('required arguments')

    crop_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the cropped data set should be written to.
        ''',
        required=True,
    )
    crop_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file.
        ''',
        required=True,
    )

    # optional
    crop_optional = crop_parser.add_argument_group('optional arguments')
    crop_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Path to image directory. By default, will be inferred.
        ''',
        default=None,
    )
    crop_optional.add_argument(
        '-p', '--padding',
        type = int,
        help = '''
            Padding for bbox cropping.
        ''',
        default=5,
    )
    crop_optional.add_argument(
        '-t', '--type',
        type=str,
        help='''
            Cropping type. When "segmentation" (default) is selected,
            only bboxes with a corresponding segmentation will be cropped.
        ''',
        choices=['segmentation', 'bbox'],
        default='segmentation'
    )

    # == sliding_window
    sliding_window_parser = utils_parsers.add_parser(
        'sliding_window',
        help = '''
            Crop images and corresponding annotation into sliding windows.
        ''',
        description = '''
            Crop images and corresponding annotation into sliding windows.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    sliding_window_required = sliding_window_parser.add_argument_group('required arguments')

    sliding_window_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the sliding-window cropped data set should be written to.
        ''',
        required=True,
    )
    sliding_window_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file for COCO annotations or
            path to a directory containing XML annotations for PVOC annotations.
        ''',
        required=True,
    )

    # optional
    sliding_window_optional = sliding_window_parser.add_argument_group('optional arguments')
    sliding_window_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Path to image directory. By default, will be inferred.
        ''',
        default = None,
    )
    sliding_window_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If "auto", will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto',
    )
    sliding_window_optional.add_argument(
        '-s', '--window_size',
        type = int,
        nargs = '+',
        help = '''
            Sliding window size in pixel.
            If one argument is passed, quadratic windows of window_size will be generated.
            If two arguments are passed, they are interpreted as window width and height, respectively.

            "-s 500", for example, crops sliding windows of size 500*500 (w*h), while "-s 500 300" crops 
            windows of size 500*300.
        ''',
        default=[1000],
    )
    sliding_window_optional.add_argument(
        '-p', '--overlap',
        type = int,
        nargs = '+',
        help = '''
            Overlap between sliding windows.
            If one argument is passed, the same overlap is used in horizontal and vertical direction.
            If two arguments are passed, they are interpreted as overlap in horizontal and
            vertical, respectively.
        ''',
        default=[200],
    )
    sliding_window_optional.add_argument(
        '-k', '--task',
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for. Only applies to COCO
            datasets.
        ''',
        default = 'instance-segmentation',
    )
    sliding_window_optional.add_argument(
        '-m', '--img_id',
        type = int,
        help = '''
            Starting image ID for newly generated image annotations.
        ''',
        default=1,
    )
    sliding_window_optional.add_argument(
        '-b', '--obj_id',
        type = int,
        help = '''
            Starting object ID for newly generated object annotations.
        ''',
        default=1,
    )
    sliding_window_optional.add_argument(
        '-r', '--remove_empty',
        dest = 'remove_empty',
        action = 'store_true',
        help = '''
            If this flag is set, cropped images without object annotation will
            not be saved.
        '''
    )
    parser.set_defaults(remove_empty = False)
    sliding_window_optional.add_argument(
        '-c', '--remove_incomplete',
        dest = 'remove_incomplete',
        action = 'store_true',
        help = '''
            If this flag is set, object annotations that are touched (trimmed)
            by a sliding-window edge are removed from the corresponding sliding-window images. 
        '''
    )
    parser.set_defaults(remove_incomplete = False)

    # == sw_image
    sw_image_parser = utils_parsers.add_parser(
        'sw_image',
        help = '''
            Crop images into sliding windows.
        ''',
        description = '''
            Crop images into sliding windows.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    sw_image_required = sw_image_parser.add_argument_group('required arguments')

    sw_image_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the sliding-window images should be written to.
        ''',
        required=True,
    )
    sw_image_required.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Path to image directory. By default, will be inferred.
        ''',
        required = True,
    )

    # optional
    sw_image_optional = sw_image_parser.add_argument_group('optional arguments')
    sw_image_optional.add_argument(
        '-s', '--window_size',
        type = int,
        nargs = '+',
        help = '''
            Sliding window size in pixel.
            If one argument is passed, quadratic windows of window_size will be generated.
            If two arguments are passed, they are interpreted as window width and height, respectively.

            "-s 500", for example, crops sliding windows of size 500*500 (w*h), while "-s 500 300" crops 
            windows of size 500*300.
        ''',
        default=[1000],
    )
    sw_image_optional.add_argument(
        '-p', '--overlap',
        type = int,
        nargs = '+',
        help = '''
            Overlap between sliding windows.
            If one argument is passed, the same overlap is used in horizontal and vertical direction.
            If two arguments are passed, they are interpreted as overlap in horizontal and
            vertical, respectively.
        ''',
        default=[200],
    )

    # == sw_split
    sw_split_parser = utils_parsers.add_parser(
        'sw_split',
        help = '''
            Crop train-test-val-split (ginjinn split) images and corresponding annotations
            into sliding windows.
        ''',
        description = '''
            Crop train-test-val-split (ginjinn split) images and corresponding annotations
            into sliding windows.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    sw_split_required = sw_split_parser.add_argument_group('required arguments')

    sw_split_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the sliding-window cropped data sets should be written to.
        ''',
        required=True,
    )
    sw_split_required.add_argument(
        '-i', '--split_dir',
        type = str,
        help = '''
            Path to directory generated by ginjinn split.
        ''',
        required=True,
    )

    # optional
    sw_split_optional = sw_split_parser.add_argument_group('optional arguments')
    sw_split_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If 'auto', will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto',
    )
    sw_split_optional.add_argument(
        '-s', '--window_size',
        type = int,
        nargs = '+',
        help = '''
            Sliding window size in pixel.
            If one argument is passed, quadratic windows of window_size will be generated.
            If two arguments are passed, they are interpreted as window width and height, respectively.

            "-s 500", for example, crops sliding windows of size 500*500 (w*h), while "-s 500 300" crops 
            windows of size 500*300.
        ''',
        default=[1000],
    )
    sw_split_optional.add_argument(
        '-p', '--overlap',
        type = int,
        nargs = '+',
        help = '''
            Overlap between sliding windows in pixel.
            If one argument is passed, the same overlap is used in horizontal and vertical direction.
            If two arguments are passed, they are interpreted as overlap in horizontal and
            vertical, respectively.
        ''',
        default=[0.5],
    )
    sw_split_optional.add_argument(
        '-k', '--task',
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for. Only applies to COCO
            datasets.
        ''',
        default = 'instance-segmentation',
    )
    sw_split_optional.add_argument(
        '-m', '--img_id',
        type = int,
        help = '''
            Starting image ID for newly generated image annotations.
        ''',
        default=1,
    )
    sw_split_optional.add_argument(
        '-b', '--obj_id',
        type = int,
        help = '''
            Starting object ID for newly generated object annotations.
        ''',
        default=1,
    )
    sw_split_optional.add_argument(
        '-r', '--remove_empty',
        dest = 'remove_empty',
        action = 'store_true',
        help = '''
            If this flag is set, cropped images without object annotation will
            not be saved.
        '''
    )
    parser.set_defaults(remove_empty = False)
    sw_split_optional.add_argument(
        '-c', '--remove_incomplete',
        dest = 'remove_incomplete',
        action = 'store_true',
        help = '''
            If this flag is set, object annotations that are touched (trimmed)
            by a sliding-window edge are removed from the corresponding sliding-window images. 
        '''
    )
    parser.set_defaults(remove_incomplete = False)

    # == sw_merge
    sw_merge_parser = utils_parsers.add_parser(
        'sw_merge',
        help = '''
            <EXPERIMENTAL> Merge sliding-window cropped images and annotations.
            Objects will be merged only if they satisfy ALL three thresholds.
        ''',
        description = '''
            <EXPERIMENTAL> Merge sliding-window cropped images and annotations.
            Objects will be merged only if they satisfy ALL three thresholds.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    sw_merge_required = sw_merge_parser.add_argument_group('required arguments')
    sw_merge_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory. Will be created if it does not exist.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Path to directory containing the sliding-window cropped images.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to the JSON annotation file.
        ''',
        required=True,
    )
    sw_merge_required.add_argument(
        '-t', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )

    # optional
    sw_merge_optional = sw_merge_parser.add_argument_group('optional arguments')
    sw_merge_optional.add_argument(
        '-c', '--intersection_threshold',
        type = int,
        help = '''
            Absolute intersection threshold for merging in pixel.
        ''',
        default=0,
    )
    sw_merge_optional.add_argument(
        '-u', '--iou_threshold',
        type = float,
        help = '''
            Intersection over union threshold for merging in pixel.
        ''',
        default=0.5,
    )
    sw_merge_optional.add_argument(
        '-s', '--ios_threshold',
        type = float,
        help = '''
            Intersection over smaller object threshold for merging in pixel.
        ''',
        default=0.8,
    )


    # == filter
    filter_parser = utils_parsers.add_parser(
        'filter',
        help = '''
            Filter annotation categories.
        ''',
        description = '''
            Filter annotation categories.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    filter_parser_required = filter_parser.add_argument_group('required arguments')
    filter_parser_required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Output directory the filtered annotations, and optionally filtered images
            (see img_dir option), should be written to.
        ''',
        required=True,
    )
    filter_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON) or PVCO annotation directory.
        ''',
        required=True,
    )
    filter_parser_required.add_argument(
        '-f', '--filter',
        type = str,
        help = '''
            Names of categories to filter. Filtering depends on the drop parameter.
            By default, the passed categories are kept and the remaining ones are dropped.
        ''',
        action = 'append',
        required = True,
    )

    # optional
    filter_parser_optional = filter_parser.add_argument_group('optional arguments')
    filter_parser_optional.add_argument(
        '-t', '--ann_type',
        type = str,
        help = '''
            Annotation type. If "auto", will be inferred.
        ''',
        choices = ['auto', 'COCO', 'PVOC'],
        default = 'auto',
    )
    filter_parser_optional.add_argument(
        '-d', '--drop',
        action = 'store_true',
        help = '''
            Drop categories in filter instead of keeping them.
        '''
    )
    parser.set_defaults(drop = False)

    filter_parser_optional.add_argument(
        '-i', '--img_dir',
        type = str,
        help = '''
            Directory containing the annotated images. Use this parameter if you
            want to filter out images without annotation after category filtering.
        ''',
        required=False,
        default=None,
    )
    filter_parser_optional.add_argument(
        '-c', '--copy_images',
        action = 'store_true',
        help = '''
            Copy images to img_dir instead of hard linkin them.
        '''
    )
    parser.set_defaults(copy_images = False)

    # == filter_size
    filter_size_parser = utils_parsers.add_parser(
        'filter_size',
        help = '''
            Filter COCO object annotations by size.
        ''',
        description = '''
            Filter COCO object annotations by size.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    filter_size_parser_required = filter_size_parser.add_argument_group('required arguments')
    filter_size_parser_required.add_argument(
        '-o', '--out_file',
        type = str,
        help = '''
            Annotation file (JSON) the filtered annotations should be written to.
        ''',
        required=True,
    )
    filter_size_parser_required.add_argument(
        '-a', '--ann_file',
        type = str,
        help = '''
            Path to COCO annotation file (JSON).
        ''',
        required=True,
    )
    filter_size_parser_required.add_argument(
        '-d', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )

    # optional
    filter_size_parser_optional = filter_size_parser.add_argument_group(
        'optional arguments'
    )
    filter_size_parser_optional.add_argument(
        '-x', '--min_width',
        type = int,
        default = 5,
        help = '''
            Minimal total object width.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-y', '--min_height',
        type = int,
        default = 5,
        help = '''
            Minimal total object height.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-r', '--min_area',
        type = int,
        default = 25,
        help = '''
            Minimal total object area.
        '''
    )
    filter_size_parser_optional.add_argument(
        '-f', '--min_fragment_area',
        type = int,
        default = 25,
        help = '''
            Minimal object fragment area.
        '''
    )

    # == visualize
    visualize_parser = utils_parsers.add_parser(
        'visualize',
        help = '''
            Visualize object annotations on images.
        ''',
        description = '''
            Visualize object annotations on images.
        ''',
        aliases=['vis'],
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
            Path to COCO annotation file (JSON) or PVCO annotation directory.
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
            Directory containing (potentially a subset) of the annotated images.
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

    # == count
    count_parser = utils_parsers.add_parser(
        'count',
        help = '''
            Count objects per category for each image annotation.
        ''',
        description = '''
            Count objects per category for each image annotation.
        ''',
    )

    # required
    count_parser_required = count_parser.add_argument_group('required arguments')
    count_parser_required.add_argument(
        '-a', '--ann_path',
        type = str,
        help = '''
            Path to COCO annotation file (JSON).
        ''',
        required=True,
    )
    count_parser_required.add_argument(
        '-o', '--out_file',
        type = str,
        help = '''
            File (CSV) the category counts should be written to.
        ''',
        required=True,
    )

    # == other utils
    # ...

    return parser

# Note: It is a deliberate decision not to subclass argparse.ArgumentParser.
#       It might be preferable to work with composition instead of inheritance,
#       since it might be desirable to include postprocessing steps after argparse
#       parsing.
class GinjinnArgumentParser():
    '''GinjinnArgumentParser

    Class for setting up and handling commandline arguments.
    '''

    _description = '''
        GinJinn is a framework for simplifying the setup, training, evaluation,
        and deployment of object detection models.
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=self._description,
        )
        self.parser.add_argument(
            '-d', '--debug',
            help='Debug mode',
            action='store_true',
        )

        self._subparsers = self.parser.add_subparsers(
            dest='subcommand',
            help='GinJinn subcommands.'
        )
        self._init_subparsers()

    def parse_args(self, args=None, namespace=None):
        '''parse_args
        Parses the commandline arguments and returns them in argparse
        format.

        Parameters
        ----------
        args
            List of strings to parse. If None, the strings are taken from sys.argv.
        namespace
            An object to take the attributes. The default is a new empty argparse Namespace object.

        Returns
        -------
        args
            Parsed argparse arguments
        '''

        return self.parser.parse_args(args=args, namespace=namespace)

    def _init_subparsers(self):
        '''_init_subparsers

        Initilialize parsers for GinJinn subcommands.
        '''

        _setup_new_parser(self._subparsers)
        _setup_train_parser(self._subparsers)
        _setup_evaluate_parser(self._subparsers)
        _setup_predict_parser(self._subparsers)
        _setup_split_parser(self._subparsers)
        _setup_simulate_parser(self._subparsers)
        _setup_info_parser(self._subparsers)
        _setup_utils_parser(self._subparsers)

        # TODO: implement
