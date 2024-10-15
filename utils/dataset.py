from torch.utils.data import Dataset
import os


class customDataset(Dataset):
    def __init__(self,data_path):
        super(customDataset,self).__init__()
        self.rootdir = data_path
        self.classes = os.listdir(data_path)

    def __getitem__(self, item):
        pass

    def __len__(self):
        sum = 0
        for cls in self.classes:
            temppath = os.path.join(self.rootdir,cls)
            sum += len(os.listdir(temppath))
        return sum

if __name__ == '__main__':
    path = '/Users/lxw/07/conditionGAN/data/animefaces'
    dataset = customDataset(path)
    print(dataset.__len__())