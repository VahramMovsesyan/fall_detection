import torch
import numpy as np
import torch.nn as nn

device = 'cude' if torch.cuda.is_available() else 'cpu'

class FallDetecion():
    def __init__(self):
        self.cache = torch.zeros(108)

    def __call__(self, skeleton_cache):
        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''

        raise NotImplementedError
        return isFall, fallScore 
    

# Load skeleton data
def load_skeleton_data(file_path):
    return np.load(file_path)

# Fill missing points using interpolation
def fill_missing_points(skeleton_data):
    # Fill missing points with NaNs
    skeleton_data[skeleton_data <= 0] = np.nan
    try:
        for i in range(108):
            arr = skeleton_data[i,...,1]
            indices = np.arange(len(arr))
            known_indices = indices[~np.isnan(arr)]
            arr = np.interp(indices, known_indices, arr[known_indices])
            skeleton_data[i,...,1] = arr

    # uses for skelethon_data_2
    except ValueError:
        for i in range(17):
            arr = skeleton_data[...,i,1]
            indices = np.arange(len(arr))
            known_indices = indices[~np.isnan(arr)]
            arr = np.interp(indices, known_indices, arr[known_indices])
            skeleton_data[...,i,1] = arr

            arr = skeleton_data[...,i,0]
            indices = np.arange(len(arr))
            known_indices = indices[~np.isnan(arr)]
            arr = np.interp(indices, known_indices, arr[known_indices])
            skeleton_data[...,i,0] = arr

        
    return skeleton_data


# load data
skeleton_data_1 = load_skeleton_data('./data/skeleton_1.npy')
skeleton_data_2 = load_skeleton_data('./data/skeleton_2.npy')
skeleton_data_3 = load_skeleton_data('./data/skeleton_3.npy')

# data preprocessing 
skeleton_data_1 = fill_missing_points(skeleton_data_1)
skeleton_data_2 = fill_missing_points(skeleton_data_2)
skeleton_data_3 = fill_missing_points(skeleton_data_3)

