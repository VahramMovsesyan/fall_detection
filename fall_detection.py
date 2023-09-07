import numpy as np
import torch
from matplotlib import pyplot as plt

# device = 'cude' if torch.cuda.is_available() else 'cpu'


class FallDetecion():
    def __init__(self, threshold: float | int = 5.0):
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
        self.vec_pairs =[(18,17),(17,15),(19,20),(19,16),(11,13),(6,5),(21,23),(22,23)]


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
        skeleton_cache = self.fill_missing_points(skeleton_cache)


        angles = np.array([0])
        x = [i for i in range(1617)] # for plot
        frm_angle = [] # for experiment
        # print(angles,1)
        for i in range(90):
            for y in range(i, 18+i):
                current_frm_vec, next_frm_vec = self.to_vector(skeleton_cache[y, :]), self.to_vector(skeleton_cache[y+1, :])
                # print(current_frm_vec, next_frm_vec, sep='\n',end='\n======================================================\n')
                
                current_frm_angle, next_frm_angle = self.angle_calculator(current_frm_vec), self.angle_calculator(next_frm_vec)
                # print(current_frm_angle, next_frm_angle, sep='\n',end='\n======================================================\n')
                frame_angle = np.around(np.sum(np.around(np.sum(next_frm_angle - current_frm_angle) / 8, 5))/18, 5) 
                if angles:
                    frm_angle.append(frame_angle) # for experiment
                    angles = np.around((angles + frame_angle) / 2, 5)
                    # print(angles,2)
                else:
                    angles = frame_angle
                    # print(angles,3)
        plt.plot(x,frm_angle)
        plt.xlabel("X-axis Label")
        plt.ylabel("Y-axis Label")
        plt.title("Simple Line Plot")
        plt.show()
        
        isFall = min(frm_angle) < self.threshold
        fallScore = f'Fall Score = {torch.round(torch.sigmoid(torch.sum(torch.tensor(frm_angle, dtype=torch.float32))),decimals=4)}'
        return isFall, fallScore


    # Fill missing points using interpolation
    def fill_missing_points(self,skeleton_data: np.ndarray) -> np.array:

        # Fill missing points with NaNs
        skeleton_data[skeleton_data <= 0] = np.nan
        try:
            for i in range(108):
                arr = skeleton_data[i, ..., 1]
                indices = np.arange(len(arr))
                known_indices = indices[~np.isnan(arr)]
                arr = np.interp(indices, known_indices, arr[known_indices])
                skeleton_data[i, ..., 1] = arr

        # uses for skelethon_data_2
        except ValueError:
            for i in range(17):
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

        
        # Add vertical vector
        vectors = np.array(vectors)  # 1d array -> 2d array
        # print(vectors.shape) # (23,2)
        return vectors

    def angle_calculator(self, vectors_arr: np.ndarray) -> np.ndarray:
        
        angles = []
        for pair in self.specified_connections:
            vector_1, vector_2 = vectors_arr[pair[0]], vectors_arr[pair[1]]

            dot = np.dot(vector_1, vector_2)

            magnitude1 = np.linalg.norm(vector_1)
            magnitude2 = np.linalg.norm(vector_2)

            angle_rad = np.arccos(dot / (magnitude1 * magnitude2))
            angles.append(np.degrees(angle_rad))

        return np.array(angles)
    
       

# load data
skeleton_data_1 = np.load('./data/skeleton_1.npy')
skeleton_data_2 = np.load('./data/skeleton_2.npy')
skeleton_data_3 = np.load('./data/skeleton_3.npy')


fall_obj = FallDetecion(threshold=-1.0)
print(fall_obj(skeleton_data_3))
