from torch.utils.data import Dataset
from PIL import Image
import os


class CAPTCHADataset(Dataset):

    def __init__(self, data_dir, images_metadata, transform=None):
        self.data_dir = data_dir
        self.images_metadata = images_metadata
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.images_metadata[index]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # image = transform(image)
        label = image_name.split(".")[0]
        return image, label

    def __len__(self):
        return len(self.images_metadata)
