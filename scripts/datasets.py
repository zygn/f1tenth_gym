import os, torch, cv2, pickle, math
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class SteerDataset(Dataset):
    def __init__(self, folderpath, transforms=None):
        """
        folderpath: String of directory containing pkl files
        Each pkl should be a dictionary:
        {
            "obs": observation,
            "action": action
        }
        """
        super(SteerDataset, self).__init__()
        self.pkl_list = os.listdir(folderpath)
        self.folderpath = folderpath

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        """
        Returns tuple (img- C x H x W Tensor, angle-float 1-Tensor)
        """
        pkl_name = self.pkl_list[idx]
        with open(os.path.join(self.folderpath, pkl_name), 'rb') as f:
            pkl_dict = pickle.load(f)

        obs = pkl_dict.get("obs")
        action = pkl_dict.get("action")
        cv_img = obs.get("img")[:, :, :3]
        cv_img = cv2.resize(cv_img, (0, 0), fx=0.5, fy=0.5)
        ts_angle = torch.Tensor([action.get("angle") * 180.0/math.pi]).float()
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        return {"img":ts_img, "angle":ts_angle}