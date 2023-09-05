import torch
import numpy as np

# device = 'cude' if torch.cuda.is_available() else 'cpu'

class FallDetecion():
    def __init__(self):
        pass

    def angle_calculator(self,vec1,vec2):
        self.dot = np.dot(vec1,vec2)

        self.magnitude1 = np.linalg.norm(vec1)
        self.magnitude2 = np.linalg.norm(vec2)

        self.angle_rad = np.arccos(self.dot / (self.magnitude1 * self.magnitude2))
        return  np.degrees(self.angle_rad)

    def to_vector(self,data):
        # Define connections between points
        connections = {
            0: [1, 2],
            1: [2, 3],
            2: [4],
            3: [5],
            4: [6],
            5: [6, 7, 11],
            6: [8, 12],
            7: [9],
            8: [10],
            9: [],
            10: [],
            11: [12, 13],
            12: [14],
            13: [15],
            14: [16],
            15: [],
            16: []
        }

        # Calculate the specific 19 vectors based on connections
        specific_vectors = []

        for source_point, connected_points in connections.items():
            for target_point in connected_points:
                specific_vectors.append(data[target_point] - data[source_point])

        # Display the specific 19 vectors
        specific_vectors = np.array(specific_vectors)
        print(specific_vectors)
        return specific_vectors

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
        point_data = skeleton_cache[0,:]
        self.vectors = self.to_vector(point_data)
            
        # self.dot_product = self.angle_calculator(x,y)
        # print(self.dot_product)
        #return isFall, fallScore 
    

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


fall_obj = FallDetecion()
fall_obj(skeleton_data_1)
