''' Module for the ginjinn new subcommand
'''

import shutil
import sys
import os
import re

import pkg_resources

from ginjinn.utils import confirmation_cancel
from ginjinn.utils.utils import get_dstype

TYPE_RE = re.compile(r'(type:\s+")\w+(")')
TRAINING_RE = re.compile(
    r'^(\s*training:\s*)(annotation_path:\s+").+("\s*)(image_path:\s*").+(")\s*$',
    re.MULTILINE
)
VALIDATION_RE = re.compile(
    r'^(\s*validation:\s*)(annotation_path:\s+").+("\s*)(image_path:\s*").+(")\s*$',
    re.MULTILINE
)
TEST_RE = re.compile(
    r'^(\s*test:\s*)(annotation_path:\s+").+("\s*)(image_path:\s*").+(")\s*$',
    re.MULTILINE
)

RE_MAP = {
    'train': TRAINING_RE,
    'val': VALIDATION_RE,
    'test': TEST_RE,
}

def ginjinn_new(args):
    '''ginjinn_new

    GinJinn new command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn new
        subcommand.
    '''

    template_dir = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/templates',
    )
    template_path = os.path.join(
        template_dir,
        ('adv_' if args.advanced else '') + args.template
    )


    project_dir = args.project_dir
    if os.path.exists(project_dir):
        if confirmation_cancel(
            f'\nDirectory "{project_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{project_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(project_dir)
        else:
            sys.exit()

    os.mkdir(project_dir)
    os.mkdir(os.path.join(project_dir, 'outputs'))

    with open(template_path) as cfg_template_file:
        config_str = cfg_template_file.read()
    config_str = config_str.replace('"ENTER PROJECT DIR HERE"', f'{os.path.abspath(project_dir)}')

    # update config if data_dir is passed
    data_dir = args.data_dir
    if not data_dir is None:
        print(f'Searching for dataset in "{data_dir}" ...')
        data_dir_contents = os.listdir(data_dir)

        if 'train' in data_dir_contents:
            ds_types = []
            print('... found "train"')
            print('... expecting a ginjinn split dataset')
            for ds_name in ['train', 'val', 'test']:
                if not ds_name in data_dir_contents:
                    print(f'... could not find "{ds_name}"')
                    config_str = re.sub(
                        RE_MAP[ds_name],
                            r'#\1#\2\3#\4\5',
                            config_str
                    )
                    print(f'\t... commented out {ds_name} dataset.')
                    continue

                print(f'... found "{ds_name}"')
                ds_path = os.path.join(data_dir, ds_name)
                if not os.path.isdir(ds_path):
                    print(
                        f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                        f'("{ds_name}" is not a directory)'
                    )
                    sys.exit()

                images_path = os.path.join(ds_path, 'images')
                if not os.path.isdir(images_path):
                    print(
                        f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                        f'("{ds_name}/images" is not a directory)'
                    )
                    sys.exit()

                ds_dir_contents = os.listdir(ds_path)
                # COCO
                if 'annotations.json' in ds_dir_contents:
                    print('\t... found "annotations.json"')
                    print('\t... expecting a COCO dataset')
                    annotations_path = os.path.join(ds_path, 'annotations.json')
                    if not os.path.isfile(annotations_path):
                        print(
                            f'ERROR: "{data_dir}/{ds_name}" is not a valid data directory. ' +\
                            '(expecting COCO dataset but "annotations.json" is not a file)'
                        )
                        sys.exit()

                    ann_path_rel = os.path.relpath(annotations_path, project_dir)
                    img_path_rel = os.path.relpath(images_path, project_dir)
                    config_str = re.sub(
                        RE_MAP[ds_name],
                        f'\\1\\2{ann_path_rel}\\3\\4{img_path_rel}\\5',
                        config_str
                    )
                    print(f'\tFound COCO dataset and initialized it as {ds_name} data.')
                    ds_types.append('COCO')

                # PVOC
                elif 'annotations' in ds_dir_contents:
                    print('\t... found "annotations"')
                    print('\t... expecting a PVOC dataset')

                    annotations_path = os.path.join(ds_path, 'annotations')
                    if not os.path.isdir(annotations_path):
                        print(
                            f'ERROR: "{data_dir}/{ds_name}" is not a valid data directory. ' +\
                            '(expecting PVOC dataset but "annotations" is not a directory)'
                        )
                        sys.exit()

                    ann_path_rel = os.path.relpath(annotations_path, project_dir)
                    img_path_rel = os.path.relpath(images_path, project_dir)
                    config_str = re.sub(
                        RE_MAP[ds_name],
                        f'\\1\\2{ann_path_rel}\\3\\4{img_path_rel}\\5',
                        config_str
                    )
                    print(f'\tFound PVOC dataset and initialized it as {ds_name} data.')
                    ds_types.append('PVOC')

            if len(set(ds_types)) > 1:
                print('ERROR: incompatible dataset types found.')
                sys.exit()
            config_str = re.sub(TYPE_RE, f'\\1{ds_types[0]}\\2', config_str)

        elif 'images' in data_dir_contents:
            print('... found "images"')
            images_path = os.path.join(data_dir, 'images')
            if not os.path.isdir(images_path):
                print(
                    f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                    '("images" is not a directory)'
                )
                sys.exit()
            # COCO
            if 'annotations.json' in data_dir_contents:
                print('... found "annotations.json"')
                print('... expecting a COCO dataset')
                annotations_path = os.path.join(data_dir, 'annotations.json')
                if not os.path.isfile(annotations_path):
                    print(
                        f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                        '(expecting COCO dataset but "annotations.json" is not a file)'
                    )
                    sys.exit()


                ann_path_rel = os.path.relpath(annotations_path, project_dir)
                img_path_rel = os.path.relpath(images_path, project_dir)
                config_str = re.sub(TYPE_RE, r'\1COCO\2', config_str)
                config_str = re.sub(
                    TRAINING_RE,
                    f'\\1\\2{ann_path_rel}\\3\\4{img_path_rel}\\5',
                    config_str
                )
                print('Found COCO dataset and initialized it as training data.')

            # PVOC
            elif 'annotations' in data_dir_contents:
                print('... found "annotations"')
                print('... expecting a PVOC dataset')

                annotations_path = os.path.join(data_dir, 'annotations')
                if not os.path.isdir(annotations_path):
                    print(
                        f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                        '(expecting COCO dataset but "annotations" is not a directory)'
                    )
                    sys.exit()

                ann_path_rel = os.path.relpath(annotations_path, project_dir)
                img_path_rel = os.path.relpath(images_path, project_dir)
                config_str = re.sub(TYPE_RE, r'\1PVOC\2', config_str)
                config_str = re.sub(
                    TRAINING_RE,
                    f'\\1\\2{ann_path_rel}\\3\\4{img_path_rel}\\5',
                    config_str
                )
                print('Found PVOC dataset and initialized it as training data.')

            config_str = re.sub(
                VALIDATION_RE,
                r'#\1#\2\3#\4\5',
                config_str
            )
            config_str = re.sub(
                TEST_RE,
                r'#\1#\2\3#\4\5',
                config_str
            )
            print('Commented out validation and test datasets.')

        else:
            print(
                f'ERROR: "{data_dir}" is not a valid data directory. ' +\
                '(no valid dataset found)'
            )
            sys.exit()

    config_path = os.path.join(project_dir, 'ginjinn_config.yaml')
    with open(config_path, 'w') as cfg_file:
        cfg_file.write(config_str)

    print(f'Initialized GinJinn project at "{project_dir}".')
