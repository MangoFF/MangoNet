import os
import sys
sys.path.append("...")
import inspect
from functools import partial
import torch.optim as optim
import  torchvision
from torchvision.transforms import ColorJitter,ToTensor,Normalize,RandomVerticalFlip,RandomHorizontalFlip,RandomAffine
import torchvision.datasets as datasets
import torch.optim.lr_scheduler as lr_scheduler
from midloss.train import main
from midloss.model.mangoNet import resnet34
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
    train_img_transforms = [] + val_img_transforms
    tensor_transforms = [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    epochs =300

    #使用sequence 不再使用随机取样，此行无用
    train_iterations = 20_000

    batch_size = 64
    workers = 5
    pretrained = True

    #按照densenet里面训训练的办法
    #采用SDG，lr=0.1,momentum=0.9, weight_decay=1e-4
    #学习率在总epochs的50% 和75%都衰减到原来的0.1
    optimizer = partial(optim.SGD, lr=0.1,momentum=0.9, weight_decay=1e-4)
    scheduler = partial(lr_scheduler.MultiStepLR,  milestones=[150, 225],  gamma=0.1)

    model = partial(resnet34,num_classes=10)
    exp_name = "resnet34_mangoNet"  # os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('checkpoints/CIFAR10', exp_name)
    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)

    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, train_img_transforms=train_img_transforms,
         val_img_transforms=val_img_transforms, tensor_transforms=tensor_transforms, epochs=epochs,
         train_iterations=train_iterations, batch_size=batch_size, workers=workers, optimizer=optimizer,
         scheduler=scheduler, pretrained=pretrained, model=model,seed=2022)

    # os.system('sudo shutdown')
