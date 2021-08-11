'''Commandline helpers
'''

import os
from os import listdir
from os.path import join, exists, isdir, isfile, basename, splitext
import sys
import shutil
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import tqdm

def prepare_out_dir(
    out_dir : str
):
    '''prepare_out_dir

    Make sure that out_dir exists.
    Can exit.

    Parameters
    ----------
    out_dir : str
        Output directory.
    '''
    from ginjinn.utils import confirmation_cancel

    if exists(out_dir):
        msg = f'Output directory "{out_dir}" already exists. ' +\
            f'Do you want to overwrite it? WARNING: this will remove "{out_dir}" directory ' +\
            'and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(out_dir)
            print(f'Removed directory "{out_dir}".')
            os.mkdir(out_dir)
        else:
            sys.exit(1)
    else:
        os.mkdir(out_dir)


class AnnotationType(Enum):
    '''AnnotationType'''
    COCO = 0
    PVOC = 1

class InvalidAnnotationPath(Exception):
    '''InvalidAnnotationPath'''

def get_annotation_type(ann_path: str) -> AnnotationType:
    '''get_annotation_type

    Get annotation type (COCO or PVOC) of ann_path.

    Parameters
    ----------
    ann_path : str
        Path to JSON file for COCO or a folder for PVOC.

    Returns
    -------
    AnnotationType
        Annotation type. AnnotationType.COCO or AnnotationType.PVOC.

    Raises
    ------
    InvalidAnnotationPath
        Raised if ann_path is not a valid annotation path.
    '''
    if isfile(ann_path):
        return AnnotationType.COCO
    elif isdir(ann_path):
        return AnnotationType.PVOC

    msg = f'"{ann_path}" is not a valid annotation path.'
    raise InvalidAnnotationPath(msg)

def check_ann_path(
    ann_path : str,
) -> Optional[AnnotationType]:
    '''check_ann_path

    Check if ann_path is a valid annotation path and return the annotation type.
    Can exit.

    Parameters
    ----------
    ann_path : str
        Path to COCO annotation file or PVOC annotation directory

    Returns
    -------
    Optional[AnnotationType]
        Type of annotation.
    '''

    try:
        ann_type = get_annotation_type(ann_path)
    except InvalidAnnotationPath:
        print(f'ERROR: Could not find a valid annotation at "{ann_path}".')
        sys.exit(1)

    return ann_type

def check_image_dir(
    image_dir : str,
):
    '''check_image_dir

    Check if image_dir is a valid image directory.
    Can exit.

    Parameters
    ----------
    image_dir : str
        Image directory.
    '''

    from ginjinn.utils import get_image_files

    if not exists(image_dir):
        print(f'ERROR: "{image_dir}" does not exist.')
        sys.exit(1)
    elif len(get_image_files(image_dir)) < 1:
        print(f'ERROR: "{image_dir}" does not contain any image files.')
        sys.exit(1)


def check_coco_dataset_dir(
    dataset_dir: str,
):
    '''check_coco_dataset_dir

    Check COCO dataset directory validity. Can exit.

    Parameters
    ----------
    dataset_dir : str
        COCO dataset directory.
    '''

    ann_path = join(dataset_dir, 'annotations.json')
    if not exists(ann_path):
        print(
            'ERROR: Could not find annotations.json. '
            'Not a valid COCO dataset.'
        )
        sys.exit(1)
    elif isdir(ann_path):
        print(
            'ERROR: annotations.json is a directory instead of a file. '
            'Not a valid COCO dataset.'
        )
        sys.exit(1)

    image_dir = join(dataset_dir, 'images')

    from ginjinn.utils import get_image_files
    if not exists(image_dir):
        print('ERROR: could not find images directory. Not a valid COCO dataset.')
        sys.exit(1)
    elif not isdir(image_dir):
        print('ERROR: images is no directory. Not a valid COCO dataset.')
        sys.exit(1)
    elif len(get_image_files(image_dir)) < 1:
        print('ERROR: could not find any images in images directory. Not a valid COCO dataset.')
        sys.exit(1)

def check_pvoc_dataset_dir(
    dataset_dir: str,
):
    '''check_pvoc_dataset_dir

    Check PVOC dataset directory validity. Can exit.

    Parameters
    ----------
    dataset_dir : str
        PVOC dataset directory.
    '''

    ann_dir = join(dataset_dir, 'annotations')
    if not exists(ann_dir):
        print(
            'ERROR: Could not find annotations directory. '
            'Not a valid PVOC dataset.'
        )
        sys.exit(1)
    elif not isdir(ann_dir):
        print('ERROR: annotations is no directory. Not a valid PVOC dataset.')
        sys.exit(1)

    image_dir = join(dataset_dir, 'images')

    from ginjinn.utils import get_image_files
    image_files = get_image_files(image_dir)
    if not exists(image_dir):
        print('ERROR: could not find images directory. Not a valid PVOC dataset.')
        sys.exit(1)
    elif not isdir(image_dir):
        print('ERROR: images is no directory. Not a valid PVOC dataset.')
        sys.exit(1)
    elif len(image_files) < 1:
        print('ERROR: could not find any images in images directory. Not a valid PVOC dataset.')
        sys.exit(1)

    annotation_files = [f for f in listdir(ann_dir) if f.endswith('.xml') or f.endswith('.XML')]

    if len(annotation_files) != len(image_files):
        print(
            'ERROR: unequal number of annotations and image files '
            f'(#annotations: {len(annotation_files)}; #images {len(image_files)}). '
            'Not a valid PVOC dataset.'
        )
        sys.exit(1)

    img_basenames = sorted([splitext(basename(f))[0] for f in image_files])
    ann_basenames = sorted([splitext(basename(f))[0] for f in annotation_files])
    matching_basenames = [a == i for a, i in zip(ann_basenames, img_basenames)]
    if not all(matching_basenames):
        print('ERROR: annotations and images are not compatible. Not a valid PVOC dataset.')
        sys.exit(1)


class DatasetType(Enum):
    '''DatasetType'''
    simple = 0
    split = 1

def check_dataset_dir(
    dataset_dir: str,
) -> Tuple[DatasetType, AnnotationType, Optional[Tuple[str]]]:
    '''check_dataset_dir

    Check dataset directory and return dataset information.
    Can exit.

    Parameters
    ----------
    dataset_dir : str
        Dataset directory

    Returns
    -------
    Tuple[DatasetType, AnnotationType]
        Dataset information: Dataset type, annotation type
    '''

    if not exists(dataset_dir):
        print(f'ERROR: could not find dataset directory "{dataset_dir}".')
        sys.exit(1)
    elif not isdir(dataset_dir):
        print(f'ERROR: "{dataset_dir}" is no directory.')
        sys.exit(1)

    # is COCO dataset
    if exists(join(dataset_dir, 'annotations.json')):
        print('Found annotations.json. Looking for COCO dataset ...')
        check_coco_dataset_dir(dataset_dir)
        print('Found COCO dataset.')

        return (DatasetType.simple, AnnotationType.COCO, None)

    # is PVOC dataset
    elif exists(join(dataset_dir, 'annotations')):
        print('Found annotations directory. Looking for PVOC dataset ...')
        check_pvoc_dataset_dir(dataset_dir)
        print('Found PVOC dataset.')

        return (DatasetType.simple, AnnotationType.PVOC, None)

    # is split dataset
    elif exists(join(dataset_dir, 'train')):
        print('Found train directory. Looking for split dataset ...')
        train_dir = join(dataset_dir, 'train')

        splits = []

        # is COCO
        if exists(join(train_dir, 'annotations.json')):
            print('Found annotations.json in train. Looking for COCO dataset ...')
            check_coco_dataset_dir(train_dir)
            print('train is a COCO dataset. Looking for val and test COCO dataset directories ...')
            splits.append('train')

            val_dir = join(dataset_dir, 'val')
            if isdir(val_dir):
                print('Found val directory. Checking val dataset ...')
                check_coco_dataset_dir(val_dir)
                print('val is a valid COCO dataset.')
                splits.append('val')
            else:
                print('Could not find val directory.')

            test_dir = join(dataset_dir, 'test')
            if isdir(test_dir):
                print('Found test directory. Checking test dataset ...')
                check_coco_dataset_dir(test_dir)
                print('test is a valid COCO dataset.')
                splits.append('test')
            else:
                print('Could not find test directory.')

            return (DatasetType.split, AnnotationType.COCO, tuple(splits))

        # is PVOC
        elif exists(join(train_dir, 'annotations')):
            print('Found annotations in train. Looking for PVOC dataset ...')
            check_pvoc_dataset_dir(train_dir)
            print('train is a PVOC dataset. Looking for val and test PVOC dataset directories ...')
            splits.append('train')

            val_dir = join(dataset_dir, 'val')
            if isdir(val_dir):
                print('Found val directory. Checking val dataset ...')
                check_pvoc_dataset_dir(val_dir)
                print('val is a valid PVOC dataset.')
                splits.append('val')
            else:
                print('Could not find val directory.')

            test_dir = join(dataset_dir, 'test')
            if isdir(test_dir):
                print('Found test directory. Checking test dataset ...')
                check_pvoc_dataset_dir(test_dir)
                print('test is a valid PVOC dataset.')
                splits.append('test')
            else:
                print('Could not find test directory.')

            return (DatasetType.split, AnnotationType.PVOC, tuple(splits))

        # is invalid
        else:
            print(
                f'ERROR: could not find train directory in "{dataset_dir}". '
                'Not a valid split directory.'
            )
            sys.exit(1)

    else:
        print(f'ERROR: Could not identify any valid dataset in "{dataset_dir}".')
        sys.exit(1)

class MultistepProgressBars:
    '''MultistepProgressBar

    Parameters
    ----------
    totals : Union[List[int], int]
        Total number of iterations for each single progress bar.
    descs : Optional[Union[List[str], str]], optional
        Descriptions for each single progress bar.
    units : Union[List[str], str], optional
        Units for each single progress bar, by default 'it'
    minintervals : Union[List[float], float], optional
        Minimum progress update interval for each single progress bar, by default 0.1
    maxintervals : Union[List[float], float], optional
        Maximum progress update interval for each single progress bar, by default 10
    ncols : Optional[Union[List[int], int]], optional
        Progress bar width for each single progress bar, by default None
    '''
    def __init__(
        self,
        totals: Union[List[int], int],
        descs: Optional[Union[List[str], str]] = None,
        units: Union[List[str], str] = 'it',
        minintervals: Union[List[float], float] = 0.1,
        maxintervals: Union[List[float], float] = 10,
        ncols: Optional[Union[List[int], int]] = None,
    ) -> None:
        self.n_bars = len(totals) if isinstance(totals, list) else len(totals)

        self.totals = self._prepare_variable(totals, 'totals')
        self.descs = self._prepare_variable(descs, 'descs')
        self.units = self._prepare_variable(units, 'units')
        self.minintervals = self._prepare_variable(minintervals, 'minintervals')
        self.maxintervals = self._prepare_variable(maxintervals, 'maxintervals')
        self.ncols = self._prepare_variable(ncols, 'ncols')

        self._step = -1
        self._progress = [0] * self.n_bars
        self._pbars = [None] * self.n_bars

        self._next_pbar()

    def update(self, n=1):
        '''update

        Update current progress bar. If the current progress bar is full,
        the next progress bar will be initialized according to totals.
        WARNING: Overflowing progress will be IGNORED

        Parameters
        ----------
        n : int, optional
            Number of iterations to update the current progress bar, by default 1
        '''
        step = self._step

        self._progress[step] += n
        self._pbars[step].update(n)
        if self._progress[step] >= self.totals[step]:
            self._next_pbar()

    def display(self):
        '''Display all progress bars.
        '''
        for pbar in self._pbars:
            if pbar:
                pbar.display()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        step = self._step
        pbar = self._pbars[step] if step < self.n_bars else None
        if pbar:
            pbar.close()

    def _next_pbar(self):
        '''Setup next progress bar if not finished.
        '''
        step = self._step
        pbar = self._pbars[self._step] if step > -1 else None
        if pbar:
            pbar.close()

        self._step += 1
        if self._step < self.n_bars:
            self._pbars[self._step] = tqdm.tqdm(
                total=self.totals[self._step],
                desc=self.descs[self._step],
                unit=self.units[self._step],
                mininterval=self.minintervals[self._step],
                maxinterval=self.maxintervals[self._step],
                ncols=self.ncols[self._step],
            )

    def _prepare_variable(
        self,
        var: Union[Any, List[Any]],
        var_name: str,
    ) -> List[Any]:
        '''_prepare_variable

        Prepares input variables:
            - single value -> replicate single value for each progress bar
            - list of values -> make sure list has as many entries as there are progress bars.

        Parameters
        ----------
        var : Union[Any, List[Any]]
            Variable value, either a single value or a list of values.
        var_name : str
            Name of the variable. Used for error reporting.

        Returns
        -------
        List[Any]
            Variable value as list.

        Raises
        ------
        Exception
            Raised if list length does not match the number of progress bars.
        '''
        n_bars = self.n_bars

        if isinstance(var, list):
            if len(var) != n_bars:
                msg = f'Expected {n_bars} entries but got ' +\
                    f'{len(var)} entries for variable "{var_name}".'
                raise Exception(msg)
            return var

        return [var] * n_bars
