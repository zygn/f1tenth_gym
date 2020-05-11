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
        pkl_list = os.listdir(folderpath)
        self.obs_array = []
        self.action_array = []
        for pkl_name in pkl_list:
            with open(os.path.join(folderpath, pkl_name), 'rb') as f:
                pkl_dict = pickle.load(f)
                self.obs_array.append(pkl_dict["obs"])
                self.action_array.append(pkl_dict["action"])

    def __len__(self):
        return len(self.obs_array)

    def __getitem__(self, idx):
        """
        Returns tuple (img- C x H x W Tensor, angle-float 1-Tensor)
        """
        obs = self.obs_array[idx]
        action = self.action_array[idx]
        cv_img = obs.get("img")[0]
        ts_angle = torch.Tensor([action.get("angle") * 180.0/math.pi]).float()
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        return {"img":ts_img, "angle":ts_angle}
