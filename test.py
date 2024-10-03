# import os
#
# import torch
# from torch import nn, optim
# from torch.optim import lr_scheduler
# from torchvision import transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
#
# from train_model import train_model
#
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]),
#     'val': transforms.Compose([
#         transforms.ToTensor(),
#     ]),
# }
# data_dir = '/home/zmz/datasets/tiny-imagenet-200/'
# # data_dir = 'tiny-imagenet-200'
#
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
#                                              shuffle=True, num_workers=64)
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# # class_names = image_datasets['train'].classes
#
#
# #%%
# #Load Resnet18
# model_ft = models.resnet18()
# #Finetune Final few layers to adjust for tiny imagenet input
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 200)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_ft = model_ft.to(device)
# #Multi GPU
# model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])
#
# #Loss Function
# criterion = nn.CrossEntropyLoss()
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# #%%
# #Train
# model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=200)