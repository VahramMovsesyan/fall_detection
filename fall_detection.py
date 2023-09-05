import torch
import numpy as np

# device = 'cude' if torch.cuda.is_available() else 'cpu'

class FallDetecion():
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        # Used in to_vector function
        self.specified_connections = [
            (0, 1), (0, 2), (1, 2), (1, 3), (3, 5),(5,11),
            (5, 7), (7, 9), (2, 4), (4, 6), (6, 17),
            (6, 8), (8, 10), (6, 12), (5, 17), (11, 18),
            (18, 12), (11, 13),(13, 15), (12, 14), (14, 16),
            (0, 17), (0, 18)
        ]
        self.vec_pairs = [(18,17),(17,15),(19,20),(19,16),(11,12),(11,13),(6,5),(6,7),(21,23),(22,23)]  # Used in angle_calculator function
        

    def to_vector(self,keypoints):

        # Add 3 point
        point_17 = np.array([[(keypoints[6][0] + keypoints[5][0]) / 2, (keypoints[6][1] + keypoints[5][1]) / 2]])
        point_18 = np.array([[(keypoints[12][0] + keypoints[11][0]) / 2, (keypoints[12][1] + keypoints[11][1]) / 2]])
        keypoints = np.concatenate((keypoints, point_17, point_18), axis=0)
        # print(keypoints.shape) # (19, 2)
        
        # Calculate vectors for specified connections
        vectors = []

        for connection in self.specified_connections:
            point1, point2 = connection
            vector = keypoints[point2] - keypoints[point1]
            vector = np.array(vector)
            vectors.append(vector)

        # Add vertical vector
        vertical_vector = np.array([0,1])
        vectors.append(vertical_vector)
        vectors = np.array(vectors) # 1d array -> 2d array
        return vectors
    

    def angle_calculator(self,vectors_arr: np.ndarray) -> np.ndarray: 
        angles = []

        for pair in self.vec_pairs:
            vector_1, vector_2 = vectors_arr[pair[0]], vectors_arr[pair[1]]

            dot = np.dot(vector_1,vector_2)

            magnitude1 = np.linalg.norm(vector_1)
            magnitude2 = np.linalg.norm(vector_2)

            angle_rad = np.arccos(dot / (magnitude1 * magnitude2))
            angles.append(np.degrees(angle_rad))

        return  np.array(angles)
    

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
        angles = np.array([0])
        for i in range(90):
            for y in range(i,18+i):
                current_frm_vec, next_frm_vec = self.to_vector(skeleton_cache[y,:]), self.to_vector(skeleton_cache[y+1,:])
                current_frm_angle, next_frm_angle = self.angle_calculator(current_frm_vec), self.angle_calculator(next_frm_vec)
                # print(current_frm_angle,next_frm_angle,sep='\n',end='\n======================================================\n')
                frame_angle = np.around(np.sum(np.around(np.sum(current_frm_angle - next_frm_angle) / 8,5))/18,5)
                if angles.size:
                    angles = np.around((angles + frame_angle) / 2, 5) 
                else:
                    angles = frame_angle

        isFall = angles[0] > self.threshold            
        fallScore = f'Fall Score = {None}'
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



fall_obj = FallDetecion()
print(fall_obj(skeleton_data_1))

