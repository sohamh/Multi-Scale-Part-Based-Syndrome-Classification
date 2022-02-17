import numpy as np
base='/usr/local/micapollo01/MIC/DATA/STAFF/smahdi0/tmp/'
import torch
from torch.utils.data import Dataset, DataLoader, Subset
# import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.io import loadmat


def rgb2grayscale(rgb):
    return np.float32(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))

def exclude_nans(X,g):
    N = np.sum(np.sum(np.isnan(X), axis=2), axis=1)
    E = np.where(N == 8321 * 3)
    X = np.delete(X, E,axis=0)
    g=np.delete(g, E, axis=0)
    return X,g


class FacialDataset(Dataset):
    """Face Label dataset."""

    def __init__(self, args):
        if (args.test_controls == True):
            tmp = loadmat(args.test_control_data_dir)  # data_dir)
            self.faces = np.float32(tmp["symShapes"])
            self.groups = np.float32(tmp["group_ID"])

        elif(args.test==True):
            if (args.test_val==False): # to save embeddings for test data
                tmp = loadmat(args.test_data_dir)#data_dir)
                self.faces = np.float32(tmp["symShapes"])
                self.groups = np.float32(tmp["group_ID"])
            else: # to save the embeddings for train data
                tmp = loadmat(args.data_dir)
                self.faces = np.float32(tmp["symShapes"])
                self.groups = np.float32(tmp["group_ID"])
        # self.groups = self.groups[-500:, :]
        else:
            tmp = loadmat(args.data_dir)

            self.faces = np.float32(tmp["symShapes"])
            self.groups = np.float32(tmp["group_ID"])

            self.faces=self.faces[:-500,:,:]
            self.groups=self.groups[:-500,:]

        if (args.ismodular): #if it's modular segmentation, load the modules and their corresponding labels
            seg_file = loadmat(args.seg_label)
            seg_label=seg_file["label"]
            avg_file=loadmat(args.avg_data_dir)
            self.avg_face = np.float32(avg_file["avg"])
            mask=np.float32(np.expand_dims((seg_label[args.seg_level,:]==args.seg_num),axis=1))
            self.faces = mask * self.faces + (1 - mask) * self.avg_face



        if(args.mean_normalize==True):#normalize
            avg_file=loadmat(args.avg_data_dir)
            self.avg_face = np.float32(avg_file["avg"])
            self.faces=self.faces-self.avg_face

        self.total_size=self.faces.shape[0]




    def __len__(self):
        return len(self.faces)

    def get_weights(self):
        return 1./ self.gcount

    def get_total_size(self):
        return self.total_size

    def __getitem__(self, idx):
        face = self.faces[idx, :, :]
        # face = np.squeeze(self.faces[idx,:,:])
        group = np.squeeze(self.groups[idx,:])-1
        return face, group
