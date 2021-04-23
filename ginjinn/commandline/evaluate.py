''' Module for the ginjinn evaluate subcommand.
'''

import os
import sys
import pandas as pd
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error

def write_evaluation(
    eval_res: dict,
    file_path: str,
    sep=',',
):
    '''write_evaluation

    Write evaluation results.

    Parameters
    ----------
    eval_res : dict
        Dictionary containing the evalaution results
    file_path : str
        Path the evaluation results should be written to.
    '''

    res_df = pd.DataFrame.from_dict(eval_res)
    res_df.to_csv(file_path, sep=sep)



def ginjinn_evaluate(args):
    '''ginjinn_evaluate

    GinJinn evaluate command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn evaluate
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

    # import here to reduce startup time when train is not called.
    from ginjinn.evaluation import evaluate
    from ginjinn.data_reader.load_datasets import load_test_set

    # checkpoint
    checkpoint_name = args.checkpoint
    checkpoint_file = os.path.join(
        config.project_dir, 'outputs', checkpoint_name
    )
    if not os.path.isfile(checkpoint_file):
        print(
            f'\nERROR: Checkpoint "{checkpoint_name}" (expected location: {checkpoint_file}) ' +\
            'does not exist. Please pass a valid checkpoint name.'
        )
        sys.exit(1)

    # register data set globally
    load_test_set(config)

    # evaluate
    res = evaluate(config, checkpoint_name=checkpoint_name)

    # write evaluation results
    eval_res_file = os.path.join(config.project_dir, 'evaluation.csv')
    write_evaluation(res, eval_res_file)
    print(f'Evaluation results written to "{eval_res_file}".')
