'''Module for simulation utilites
'''

import numpy as np

def polygon_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''Calculate polygon area from ordered vertex coordinates

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Returns
    -------
    np.ndarray
        Area of the polygon.
    '''
    # coordinate shift
    x_shift = x - x.mean()
    y_shift = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_shift[-1] * y_shift[0] - y_shift[-1]* x_shift[0]
    main_area = np.dot(x_shift[:-1], y_shift[1:]) - np.dot(y_shift[:-1], x_shift[1:])
    return 0.5*np.abs(main_area + correction)
