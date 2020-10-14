from typing import Union, List
import numpy as np


def apply_channel_corrections(value: Union[float, List[float]] = None,
    num_channels: int = 3,
    default_value: int = 0,
    id_str: str = "") -> np.ndarray:
    """
    Applies channel corrections so that input floats or list is converted into an np.ndarray of length num_channels
    input: input float or list of floats associated with each channel
    num_channels: # of channels for image data
    default_value: Default value if value is None, (0 for mean, 1 for std)
    id_str: Identifying ID for value

    """

    if value is None:
        return_value = np.ones((num_channels,)) * default_value
    elif type(value) is float:
        return_value = value * np.ones((num_channels,))
    else:
        return_value = np.array(value)
    if len(return_value) != num_channels:
        raise ValueError("Error! Ensure that %s has same length as num_channels: %d" %(id_str, num_channels))

    return return_value

# maybe move this to coralutils later
def normalize(input_array: np.ndarray,
    pixel_mean: np.ndarray,
    pixel_std: np.ndarray,
    reverse_normalize: bool = False) -> np.ndarray:
    """
    Applies normalization  to input array based upon mean and std
        input_array: rows x cols x n_channels array to normalize
        pixel_mean: mean value PER channel to normalize with (use apply_channel_corrections to get this if necessary)
        pixel_std: std value PER channel to normalize with (use apply_channel_corrections to get this if necessary)
        reverse_normalize: reverse normalization procedure
    """
    num_channels = input_array.shape[2]
    array = np.copy(input_array)
    if (num_channels != len(pixel_mean)):
        raise ValueError("Error! Ensure that pixel_mean has same length as num_channels for normalization: %d" %(num_channels))
    if (num_channels != len(pixel_std)):
        raise ValueError("Error! Ensure that pixel_std has same length as num_channels for normalization: %d" %(num_channels))

    for channel in range(num_channels):
        if reverse_normalize:
            array[:,:,channel] *= pixel_std[channel]
            array[:,:,channel] += pixel_mean[channel]
        else:
            array[:,:,channel] -= pixel_mean[channel]
            array[:,:,channel] /= pixel_std[channel]
    return array