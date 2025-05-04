import kagglehub
import os
from torchvision import datasets, transforms



class IntelImageClassificationDataset:
    def __init__(self, resize:tuple[int,int]=(150, 150)) -> None:
        path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        train_path = os.path.join(path, "seg_train/seg_train")
        test_path = os.path.join(path, "seg_test/seg_test")
        eval_path = os.path.join(path, "seg_pred")
        self.train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ])
        )
        self.test_dataset = datasets.ImageFolder(
            root=test_path,
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ])
        )
        self.eval_dataset = datasets.ImageFolder(
            root=eval_path,
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
            ])
        )
    
    def label(self, i:int) -> str:
        return {
            0:"buildings",
            1:"forest",
            2:"glacier",
            3:"mountain",
            4:"sea",
            5:"street"
        }[i]