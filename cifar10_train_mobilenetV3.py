import os
import sys
import inspect
from functools import partial
import torch.optim as optim
import  torchvision
import torch
from torchvision.transforms import ToTensor,Normalize,RandomHorizontalFlip,Pad,RandomCrop
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
from cvmodel.train import main
from cvmodel.model.mobilenetV3_cifar  import mobilenet_v3_small
if __name__ == '__main__':
    project_dir = os.path.dirname(inspect.getabsfile(main))

    data_dir = 'data'

    train_dataset = partial(datasets.CIFAR10,
    root=data_dir,
    train=True,
    download=True)
    val_dataset = partial(datasets.CIFAR10,
    root=data_dir,
    train=False,
    download=True)
      
    val_img_transforms = []
    train_img_transforms = [Pad(4),RandomCrop(32),RandomHorizontalFlip()] + val_img_transforms
    tensor_transforms = [ToTensor(), Normalize(mean=[125.307/255, 122.961/255, 113.8575/255],
    std=[51.5865/255, 50.847/255, 51.255/255])]

    epochs =200

    #使用Replacement=False 即使用不放回的随机取样
    train_iterations = 50000

    batch_size = 1024
    workers = 8

    #pretrain已经无用了
    pretrained = True

    #按照densenet里面训训练的办法
    #采用SDG，lr=0.1,momentum=0.9, weight_decay=1e-4
    #学习率在总epochs的50% 和75%都衰减到原来的0.1
    optimizer = partial(optim.SGD, lr=0.4,momentum=0.9, weight_decay=1e-4)
    scheduler = partial(lr_scheduler.MultiStepLR,  milestones=[125,175],  gamma=0.1)
    criterion=torch.nn.CrossEntropyLoss()
    model = partial(mobilenet_v3_small,num_classes=10)
    exp_name = "mobilenet_v3_small_cifar"  # os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('checkpoints/CIFAR10', exp_name)
    os.chdir(project_dir)      
    os.makedirs(exp_dir, exist_ok=True)
     
    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, train_img_transforms=train_img_transforms,
         val_img_transforms=val_img_transforms, tensor_transforms=tensor_transforms, epochs=epochs,
         train_iterations=train_iterations, batch_size=batch_size, workers=workers, optimizer=optimizer,criterion=criterion,
         scheduler=scheduler, pretrained=pretrained, model=model,seed=2022)

    # os.system('sudo shutdown')
