import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO
import glob

io=dataIO() 

class Train_Data(Dataset):
    def __init__(self, root_dir, mode="train", low_dose_list=["15s", "30s", "60s", "90s"]):


        self.low_dirs = low_dose_list
        self.full_dir = "180s"

        self.low_path = []
        self.full_path = []

        full_root = os.path.join(root_dir, mode, "patches", self.full_dir)

        full_files = sorted(glob.glob(os.path.join(full_root, "*.bin")))

        for full_file in full_files:
            filename = os.path.basename(full_file)

            for low_dir in self.low_dirs:
                low_file = os.path.join(root_dir, mode, "patches", low_dir, filename)
                self.low_path.append(low_file)
                self.full_path.append(full_file)

        self.length = len(self.low_path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imgL = io.load(self.low_path[idx])
        imgF = io.load(self.full_path[idx]) 
        
        imgL = torch.from_numpy(imgL).unsqueeze(0) 
        imgF = torch.from_numpy(imgF).unsqueeze(0) 

        return imgL, imgF


class Eval_Data(Dataset):
    def __init__(self, root_dir, mode="val", low_dose_list=["15s", "30s", "60s", "90s"]):
        """
        mode: "val" or "test"

        """

        self.low_dirs = low_dose_list
        self.full_dir = "180s"

        self.low_path = []
        self.full_path = []
        self.filename = []
        self.dose = []

        full_root = os.path.join(root_dir, mode, self.full_dir)

        full_files = sorted(glob.glob(os.path.join(full_root, "*.bin")))

        for full_file in full_files:
            filename = os.path.basename(full_file)

            for low_dir in self.low_dirs:
                low_file = os.path.join(root_dir, mode, low_dir, filename)

                self.low_path.append(low_file)
                self.full_path.append(full_file)
                self.filename.append(filename)
                self.dose.append(low_dir)

        self.length = len(self.low_path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imgL = io.load(self.low_path[idx])
        imgF = io.load(self.full_path[idx])

        imgL = torch.from_numpy(imgL).unsqueeze(0)
        imgF = torch.from_numpy(imgF).unsqueeze(0)

        filename = self.filename[idx]
        dose = self.dose[idx]

        return imgL, imgF, filename, dose





class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:

            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch

# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 

if __name__ == "__main__": 
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    center_dose_info = {
        "m660-1":["15s", "30s", "60s", "90s", "180s"], 
        "m660-2":["15s", "30s", "60s", "90s", "180s"], 
        "Flight":["30s", "60s", "90s", "120s"], 
        "DMI":["10s", "20s", "30s", "40s", "60s", "120s"], 
        "mCT_Adult":["15s", "30s", "45s", "60s", "90s"], 
        "mCT_Child":["15s", "30s", "45s", "60s", "90s"]
        }
    
    center_list = ["m660-1", "m660-2", "Flight", "DMI", "mCT_Adult", "mCT_Child"] 
    
    data_root = "/home/data/zhiwen/dataset/MC-NC/"
    train_loader_list = []
    # for center in center_list: 
    #     ds = Train_Data(root=data_root, center=center)
    #     train_loader_list.append(DataLoader(ds, batch_size=1, shuffle=True)) 
    
    dataset = {
        # 'train': Train_Data(root_dir=data_root, center_name="m660-1"), 
        'val': Test_Data(root_dir=data_root, center_name="m660-1", use_num=4), 
        # 'test': Test_Data(root_dir=data_root, center_name="m660-1", use_num=-1),
        
        } 
    train_loader = DataLoader(dataset['val'], batch_size=1, shuffle=True) 
    
    print("length:", len(train_loader))
    
    # valid_loader = DataLoader(Val_Data(root=data_root, center_list=center_list), batch_size=1) 
    
    # test_loader = DataLoader(Test_Data(root=data_root, center_list=center_list), batch_size=1)
    
    

    for counter, data in enumerate(tqdm(train_loader)): 
        import pdb 
        pdb.set_trace()