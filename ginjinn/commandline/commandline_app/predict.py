''' Module for the ginjinn predict subcommand.
'''

import os
import sys
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error


def ginjinn_predict(args):
    '''ginjinn_predict

    GinJinn predict command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn predict
        subcommand.
    '''

    project_dir = args.project_dir
    config_file = os.path.join(project_dir, 'ginjinn_config.yaml')

    if args.debug:
        config = GinjinnConfiguration.from_config_file(config_file)
    else:
        try:
            config = GinjinnConfiguration.from_config_file(config_file)
        except config_error.InvalidInputConfigurationError as err:
            print('\nInvalid input configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidModelConfigurationError as err:
            print('\nInvalid model configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidAugmentationConfigurationError as err:
            print('\nInvalid augmentation configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidGinjinnConfigurationError as err:
            print('\nInvalid GinJinn configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidOptionsConfigurationError as err:
            print('\nInvalid options configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidTrainingConfigurationError as err:
            print('\nInvalid training configuration:')
            print(err)
            sys.exit(1)
        except Exception as any_e:
            raise any_e

    image_path = args.image_path

    # input
    img_dir = None
    img_names = []
    if os.path.isdir(image_path):
        img_dir = image_path
    else:
        img_names = [image_path]

    # checkpoint
    checkpoint_name = args.weights_checkpoint
    checkpoint_file = os.path.join(config.project_dir, 'outputs', checkpoint_name)
    if not os.path.isfile(checkpoint_file):
        print(
            f'\nERROR: Checkpoint "{checkpoint_name}" (expected location: {checkpoint_file}) '
            + 'does not exist. Please pass a valid checkpoint name.'
        )
        sys.exit(1)

    # output
    from ginjinn.commandline.commandline_app.commandline_helpers import prepare_out_dir

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(config.project_dir, 'prediction')
    prepare_out_dir(out_dir)

    # other
    threshold = args.threshold
    padding = args.padding
    seg_refinement = args.seg_refinement
    refinement_mode = args.refinement_mode
    device = args.device
    crop = args.crop
    visualize = args.visualize

    output_options = ['COCO']
    if crop:
        output_options.append('cropped')
    if visualize:
        output_options.append('visualization')

    from ginjinn.predictor import GinjinnPredictor

    predictor = GinjinnPredictor.from_ginjinn_config(
        gj_cfg=config,
        img_dir=img_dir,
        outdir=out_dir,
        checkpoint_name=checkpoint_name,
    )
    predictor.save_scores = args.coco_scores

    from ginjinn.utils import get_image_files
    from ginjinn.commandline.commandline_app.commandline_helpers import (
        MultistepProgressBars,
    )

    totals = [len(get_image_files(img_dir))] * 2
    descs = (
        ['prediction', 'refinement + output gen.']
        if seg_refinement
        else ['prediction', 'output gen.']
    )

    with MultistepProgressBars(totals=totals, descs=descs, units='image') as pbars:
        predictor.predict(
            img_names=img_names,
            output_options=output_options,
            padding=padding,
            seg_refinement=seg_refinement,
            refinement_device=device,
            refinement_method=refinement_mode,
            threshold=threshold,
            progress_callback=pbars.update,
        )

    print(f'Predictions written to {out_dir}.')
