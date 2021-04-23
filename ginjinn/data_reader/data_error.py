'''Module for dataset errors.
'''

class IncompatibleDatasetsError(Exception):
    '''Errors due to incompatible datasets for training, test and validation.
    '''

class ImproperDatasetError(Exception):
    '''Errors due to an improper input dataset.
    '''
