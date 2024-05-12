import time

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as PILImage
import os
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from StableKeypoints import load_ldm, image2latent, init_random_noise, run_and_find_attn


def get_cifar10_dataloaders(batch_size=1, num_workers=4, resize_to=512, testset_size=20):
    transform_train = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.Imagenette(root='./data', download=False, transform=transform_train, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.Imagenette(root='./data', download=False, transform=transform_test, split='val')
    testset, _ = torch.utils.data.random_split(testset, [testset_size, len(testset) - testset_size])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def optimize_embeddings(ldm, train_dataloader, val_dataloader,
                        context=None, num_tokens=1000, device="cuda",
                        layers=[0, 1, 2, 3, 4, 5], noise_level=-1,
                        from_where=["down_cross", "mid_cross", "up_cross"], num_classes=10):
    if context is None:
        context = init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    # import torch.nn as nn
    # attention_aggregator = nn.Sequential(
    #     nn.Conv2d(1000, 32, kernel_size=3, stride=1, padding=1),  # Output: [1000, 32, 128, 128]
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [1000, 32, 64, 64]
    #
    #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: [1000, 64, 64, 64]
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [1000, 64, 32, 32]
    #
    #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: [1000, 128, 32, 32]
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [1000, 128, 16, 16]
    #
    #     nn.Flatten(),  # Flatten for the linear layer
    #     nn.Linear(128 * 16 * 16, 1024),  # Fully connected layer
    #     nn.ReLU(),
    #     nn.Linear(1024, 10)  # Classification into 10 classes
    # ).to(device)
    # attention_aggregator.requires_grad = True

    # attention_aggregator = torch.nn.Sequential(
    #     torch.nn.Linear(1000, 256),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(256, 64),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(64, 10),
    # ).to(device)

    linear_layer = torch.nn.Linear(num_tokens, num_classes).to(device)
    linear_layer.requires_grad = True

    # random project from 1000 to 10

    # optimizer = torch.optim.Adam([context], lr=1e-3)
    optimizer = torch.optim.Adam([context, linear_layer.weight, linear_layer.bias], lr=5e-3)
    # optimizer = torch.optim.Adam([
    #     {'params': context},
    #     {'params': attention_aggregator.parameters()}
    # ], lr=5e-3)
    cross_entropy = torch.nn.CrossEntropyLoss()

    # all traininalbe parameters

    dataloader_iter = iter(train_dataloader)

    for i in tqdm(range(10000)):
        try:
            images, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            images, labels = next(dataloader_iter)

        loss = 0
        # for image, label in tqdm(zip(images, labels)):
        for image, label in zip(images, labels):
            image = image.unsqueeze(0)
            label = label.to(device)

            attn_maps = run_and_find_attn(
                ldm,
                image,
                context,
                layers=layers,
                noise_level=noise_level,
                from_where=from_where,
                upsample_res=-1,
                device=device,
                controllers=controllers,
            )
            # mean on the dims 1,2
            attn_map = attn_maps[0]

            attn_map = torch.mean(attn_map, dim=(1, 2))
            output = linear_layer(attn_map)
            output = torch.nn.functional.softmax(output, dim=0)

            # attn_map = attention_aggregator(attn_map.unsqueeze(0))
            # attn_map = attn_map.squeeze(0)
            # output = torch.nn.functional.softmax(attn_map, dim=0)

            loss += cross_entropy(output, label)

        loss /= len(labels)
        loss.backward()
        if (i + 1) % len(labels) == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

        if i % 100 == 0:
            # validation
            correct = 0
            total = 0

            with torch.no_grad():
                for idx, (images, labels) in enumerate(val_dataloader):
                    # labels = labels.to(device)
                    labels = labels.to(device).unsqueeze(0)

                    attn_maps = run_and_find_attn(
                        ldm,
                        images,
                        context,
                        layers=layers,
                        noise_level=noise_level,
                        from_where=from_where,
                        upsample_res=-1,
                        device=device,
                        controllers=controllers,
                    )


                    attn_map = attn_maps[0]
                    if idx == 0:
                        plt.imshow(images[0].permute(1, 2, 0).detach().cpu())
                        plt.show()
                        plt.imshow(torch.mean(attn_map, dim=0).detach().cpu())
                        plt.show()
                        plt.imshow(attn_map[0].detach().cpu())
                        plt.show()
                    attn_maps = torch.mean(attn_map, dim=(1, 2))
                    outputs = linear_layer(attn_maps)

                    # attn_maps = attention_aggregator(attn_map.unsqueeze(0))
                    # outputs = torch.nn.functional.softmax(attn_maps, dim=1)
                    # outputs = outputs.squeeze(0)

                    predicted = torch.argmax(outputs, 0)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # print(f"Loss: {loss.item()}")
            print(f"Accuracy: {100 * correct / total}")
            shuffled = torch.randperm(labels.size(0))
            # shuffled according to the labels

    return context.detach()


if __name__ == '__main__':
    trainloader, testloader = get_cifar10_dataloaders()

    ldm, controllers, num_gpus = load_ldm("cuda:0",
                                          "runwayml/stable-diffusion-v1-5",
                                          feature_upsample_res=128)

    context = optimize_embeddings(ldm, trainloader,
                                  val_dataloader=testloader, device="cuda:0")
