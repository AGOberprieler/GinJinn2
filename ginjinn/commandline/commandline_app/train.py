''' Module for the ginjinn train subcommand.
'''

import os
import sys
import glob
from ginjinn.ginjinn_config import GinjinnConfiguration
from ginjinn.utils import confirmation_cancel
import ginjinn.ginjinn_config.config_error as config_error

def write_training(
    train_res: dict,
    file_path: str,
):
    '''write_training [summary]

    Parameters
    ----------
    train_res : dict
        [description]
    file_path : str
        [description]
    '''

def ginjinn_train(args):
    '''ginjinn_train

    GinJinn train command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn train
        subcommand.
    '''

    # import here to reduce startup time when train is not called.
    from ginjinn.data_reader.load_datasets import load_train_val_sets
    from ginjinn.trainer import ValTrainer, Trainer

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

    resume = args.resume if not args.resume is None else config.options.resume

    # file cleanup
    force = args.force
    if not resume:
        outputs_dir = os.path.join(config.project_dir, 'outputs')
        metrics_file = os.path.join(outputs_dir, 'metrics.json')

        to_remove = []

        if os.path.exists(metrics_file):
            if force or confirmation_cancel(
                'WARNING: Resume is set to False but an old metrics.json file was found. ' +\
                'Should the metrics.json file be removed? Keeping it will cause ' +\
                'problems with plotting of the training metrics.\n'
            ):
                to_remove.append(metrics_file)

        events_files = glob.glob(os.path.join(outputs_dir, 'events.out.tfevents.*'))
        if len(events_files) > 0:
            if force or confirmation_cancel(
                'WARNING: Resume is set to False but old events files were found. ' +\
                'Should the events files be removed? ' +\
                'Keeping them will cause problems when inspecting ' +\
                'the training with TensorBoard.\n'
            ):
                to_remove = to_remove + events_files

        other_output_files = [
            *glob.glob(os.path.join(outputs_dir, 'model_*')),
            *glob.glob(os.path.join(outputs_dir, 'inference', '*')),
            *glob.glob(os.path.join(outputs_dir, 'last_checkpoint')),
            *glob.glob(os.path.join(outputs_dir, 'test_*')),
            *glob.glob(os.path.join(outputs_dir, 'instances_predictions.pth')),
            *glob.glob(os.path.join(outputs_dir, 'coco_instances_results.json')),
            *glob.glob(os.path.join(outputs_dir, 'metrics.pdf')),
        ]
        if len(other_output_files) > 0:
            if force or confirmation_cancel(
                'WARNING: Resume is set to False but old intermediate files were found. ' +\
                'Should the intermediate files be removed?\n'
            ):
                to_remove = to_remove + other_output_files

        for f_path in to_remove:
            os.remove(f_path)

    # register dataset(s) globally
    load_train_val_sets(config)

    # overwrite max_iter if passed as commandline argument
    if not args.n_iter is None:
        config.training.max_iter = args.n_iter

    if config.input.val:
        trainer = ValTrainer.from_ginjinn_config(config)
    else:
        trainer = Trainer.from_ginjinn_config(config)

    trainer.resume_or_load(resume=resume)
    train_res = trainer.train()

    print(train_res)
