import os
import inspect
from functools import partial
from functools import partial
import torch.optim as optim
from torchvision.transforms import ColorJitter,ToTensor,Normalize
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
from ...model.mangoNet import MangoNet
from ...test import main

if __name__ == '__main__':
    project_dir = os.path.dirname(inspect.getabsfile(main))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]  # Make sure the config and model have the same base name
    exp_dir = os.path.join('tests', exp_name)
    model = os.path.join('weights', exp_name + '.pth')
    data_dir = 'data/vocsbd'    # The dataset will be downloaded automatically
    test_dataset = partial(datasets.FashionMNIST, data_dir, 'val')
    test_dataset = partial(datasets.FashionMNIST,
                          root="data",
                          train=False,
                          download=True)
    img_transforms = []
    tensor_transforms = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, model=model, test_dataset=test_dataset, img_transforms=img_transforms,
         tensor_transforms=tensor_transforms, forced=True)
