import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class SudokuDigitDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        datasets = ['mnist', 'hoda', 'empty_cells']
        for ds in datasets:
            ds_path = os.path.join(root_dir, ds, split)
            if not os.path.exists(ds_path):
                continue
                
            for class_name in os.listdir(ds_path):
                class_dir = os.path.join(ds_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                label = int(class_name)
                for img_path in glob.glob(os.path.join(class_dir, "*.png")):
                    self.samples.append((img_path, label))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label
