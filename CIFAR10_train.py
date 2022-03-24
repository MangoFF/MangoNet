import os
import sys
sys.path.append("...")
import inspect
from functools import partial
import torch.optim as optim
from torchvision.transforms import ColorJitter,ToTensor,Normalize
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
from midloss.train import main
from torchvision.models.resnet import resnet34
if __name__ == '__main__':
    project_dir = os.path.dirname(inspect.getabsfile(main))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('checkpoints/fasion', exp_name)
    data_dir = 'data/fasion'
    train_dataset = partial(datasets.CIFAR10,
    root="data",
    train=True,
    download=True)
    val_dataset = partial(datasets.CIFAR10,
    root="data",
    train=False,
    download=True)
    val_img_transforms = []
    train_img_transforms = [] + val_img_transforms
    tensor_transforms = [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    epochs = 100
    train_iterations = 20_000
    batch_size = 64
    workers = 5
    pretrained = True
    optimizer = partial(optim.Adam, lr=1e-4, betas=(0.5, 0.999))
    scheduler = partial(lr_scheduler.StepLR, step_size=3, gamma=0.98)
    model = partial(resnet34,num_classes=10)

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, train_img_transforms=train_img_transforms,
         val_img_transforms=val_img_transforms, tensor_transforms=tensor_transforms, epochs=epochs,
         train_iterations=train_iterations, batch_size=batch_size, workers=workers, optimizer=optimizer,
         scheduler=scheduler, pretrained=pretrained, model=model,seed=2022)

    # os.system('sudo shutdown')
