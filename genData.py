# from VAE_dataset.VAE_trajectory.src.main import BATCH_SIZE
from typing import final
import numpy as np
import pandas as pd
import cv2, random
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


mydataset = pd.read_csv("../VAE_points.csv")
track_images = mydataset.image
traj_data = mydataset.trajectory_points
(imageTrain, imageTest, trajTrain, trajTest) = (train_test_split(track_images, traj_data,test_size=0.02,shuffle=False))

class makeTrack_trajectoryDataset():
    def __init__(self, path, trajectory):
        self.X = path
        self.y = trajectory
        self.num_points = 30
        # apply augmentations
    
    def generate(self):
        images = []
        trajectories = []
        n = len(self.X)
        for i in range(1, n):
            try:
                print(i)
                image = cv2.imread('../images/'+str(self.X.iloc[i]))
                image = cv2.resize(image, (200,152))
                image = image.astype('float32')
                image_ = image.transpose((2, 0, 1))
                images.append(image_)
                traj = ast.literal_eval(self.y.iloc[i])
                final_traj = traj.copy()[1:-1]
                removePts = random.sample(range(1, len(final_traj)-1), len(final_traj) - self.num_points-2)
                removePts.sort()
                for i in range(len(removePts)):
                    try:
                        final_traj.pop(removePts[i]-i)
                    except:
                        pass
                assert(len(final_traj)-32==0)
                final_traj.append(traj[-1])
                final_traj.insert(0, traj[0])
                # print(final_traj)
                # exit()
                scalingFac = np.array([800, 600])

                final_traj = np.array((final_traj/scalingFac))
                # final_traj = np.array(final_traj)
                final_traj = np.array(final_traj.flatten(order = 'F'))
                trajectories.append(final_traj)
            except:
                pass
        return np.array(images), trajectories

make_data = 1
if make_data:
    train_data = makeTrack_trajectoryDataset(imageTrain, trajTrain)
    train_data_im, train_data_tr = train_data.generate()
    test_data = makeTrack_trajectoryDataset(imageTest, trajTest)
    test_data_im, test_data_tr = test_data.generate()
    np.save("../Numpy_Dataset/test_data_im_torch.npy", test_data_im)
    np.save("../Numpy_Dataset/test_data_tr_torch.npy",  test_data_tr)
    np.save("../Numpy_Dataset/train_data_im_torch.npy", train_data_im)
    np.save("../Numpy_Dataset/train_data_tr_torch.npy", train_data_tr)

