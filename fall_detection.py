import numpy as np
import torch
from matplotlib import pyplot as plt

# device = 'cude' if torch.cuda.is_available() else 'cpu'


class FallDetecion():
    def __init__(self, threshold=5.0):
        self.threshold = threshold
        # Used in to_vector function
        self.specified_connections = [
            (0,1),(0,2),(1,2),(1,3),(3,5), 
            (5,11),(5,7),(7,9),(2,4),(4,6), 
            (6,17),(6,8),(6,12),(5,17),(11,18),(12,18),
            (11,13),(15,13),(12,14),(16,14),(0,17),(0,18),(19,20)
        ]
        # print(len(self.specified_connections)) # 23
        
        # Used in angle_calculator function. The last two pair for vertical vector
        self.vec_pairs =[(17,16),(16,14),(18,19),(18,15),(10,11),(5,4),(20,22),(21,22)]


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
        self.frame_count = skeleton_cache.shape[0]
        skeleton_cache = self.fill_missing_points(skeleton_cache)
        
        angles = 0
        frm_angle = np.empty((1))
        for i in range(self.frame_count-1):
            current_frm_vec, next_frm_vec = self.to_vector(skeleton_cache[i, :]), self.to_vector(skeleton_cache[i+1, :])
            current_frm_angle, next_frm_angle = self.angle_calculator(current_frm_vec), self.angle_calculator(next_frm_vec)
            angl_diff = abs(next_frm_angle - current_frm_angle)
            frm_angle = np.append(frm_angle,np.around(np.mean(angl_diff),5))
        
        angles = np.round(np.mean(frm_angle),5)
        isFall = angles > self.threshold
        fallScore = f'Fall Score = {angles}'
        return isFall, fallScore


    # Fill missing points using interpolation
    def fill_missing_points(self,skeleton_data: np.ndarray) -> np.array:

        # Fill missing points with NaNs
        skeleton_data[skeleton_data <= 0] = np.nan

        try:
            for i in range(self.frame_count):
                arr = skeleton_data[i, ..., 1]
                indices = np.arange(len(arr))
                known_indices = indices[~np.isnan(arr)]
                arr = np.interp(indices, known_indices, arr[known_indices])
                skeleton_data[i, :, 1] = arr
    
                arr = skeleton_data[i, :, 0]
                indices = np.arange(len(arr))
                known_indices = indices[~np.isnan(arr)]
                arr = np.interp(indices, known_indices, arr[known_indices])
                skeleton_data[i, :, 0] = arr

        # uses for skelethon_data_2
        except ValueError:
            for i in range(skeleton_data.shape[1]): # 17
                arr = skeleton_data[..., i, 1]
                indices = np.arange(len(arr))
                known_indices = indices[~np.isnan(arr)]
                arr = np.interp(indices, known_indices, arr[known_indices])
                skeleton_data[..., i, 1] = arr

                arr = skeleton_data[..., i, 0]
                indices = np.arange(len(arr))
                known_indices = indices[~np.isnan(arr)]
                arr = np.interp(indices, known_indices, arr[known_indices])
                skeleton_data[..., i, 0] = arr

        return skeleton_data
    
    # Create vectors from 17 + 4 points
    def to_vector(self, keypoints: np.ndarray) -> np.ndarray:

        # Add 4 point
        point_17 = np.array([[(keypoints[6][0] + keypoints[5][0]) / 2, (keypoints[6][1] + keypoints[5][1]) / 2]])
        point_18 = np.array([[(keypoints[12][0] + keypoints[11][0]) / 2, (keypoints[12][1] + keypoints[11][1]) / 2]])
        vertical_start = np.array([[0,-1]]) # point 19
        vertical_end = np.array([[0,0]]) # point 20
        keypoints = np.concatenate((keypoints, point_17, point_18,vertical_start,vertical_end), axis=0)
        # print(keypoints) # (21, 2)

        # Calculate vectors for specified connections
        vectors = []

        for point1, point2 in self.specified_connections:
            vector = np.array(keypoints[point2] - keypoints[point1])
            vectors.append(vector)

        
        vectors = np.array(vectors)  # 1d array -> 2d array
        # print(vectors.shape) # (23,2)
        return vectors

    # Calculate vector's angles
    def angle_calculator(self, vectors_arr: np.ndarray) -> np.ndarray:
        
        angles = []
        for pair in self.vec_pairs:
            vector_1, vector_2 = vectors_arr[pair[0]], vectors_arr[pair[1]]

            dot = np.dot(vector_1, vector_2)

            magnitude1 = np.linalg.norm(vector_1)
            magnitude2 = np.linalg.norm(vector_2)
            angle_in_radians = np.arccos(np.clip(dot / (magnitude1 * magnitude2), -1.0, 1.0))
            angles.append(np.degrees(angle_in_radians))

        return np.array(angles)
       

# load data
skeleton_data_1 = np.load('./data/skeleton_1.npy')
skeleton_data_2 = np.load('./data/skeleton_2.npy')
skeleton_data_3 = np.load('./data/skeleton_3.npy')


fall_obj = FallDetecion(threshold=6.1)

# Array contain all fall score. This for visualization
# arr = np.empty((1)) 

# Testing algorithm like mashine. Delete for loop and give SPLITTED skelethon cache as an OBJECT argument 
for i in range(90):
    print(fall_obj(skeleton_data_2[i:i+19,...]))
    # arr = np.append(arr,float(fall_obj(skeleton_data_3[i:i+19,...])[1].split(" = ")[1]))

# plt.plot(arr)
# plt.show()
