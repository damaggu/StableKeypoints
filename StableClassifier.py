import pickle
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

from StableKeypoints import load_ldm, image2latent, init_random_noise, run_and_find_attn, RandomAffineWithInverse, \
    equivariance_loss


def get_cifar10_dataloaders(batch_size=1, num_workers=4, resize_to=512, testset_size=20):
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dl = True if not os.path.exists('./data/imagenette2') else False

    trainset = torchvision.datasets.Imagenette(root='./data', download=dl, transform=transform_train, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.Imagenette(root='./data', download=dl, transform=transform_test, split='val')
    testset, _ = torch.utils.data.random_split(testset, [testset_size, len(testset) - testset_size])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


import torch.nn.functional as F


# def equivariance_loss(embeddings_initial, embeddings_transformed, transform, index):
#     embeddings_initial_prime = transform.inverse(embeddings_transformed)[index]
#     loss = F.mse_loss(embeddings_initial, embeddings_initial_prime)
#     return loss

def total_variation_loss_fn(attn_map):
    tv_loss = torch.mean(torch.abs(attn_map[:, :-1, :] - attn_map[:, 1:, :])) + \
              torch.mean(torch.abs(attn_map[:, :, :-1] - attn_map[:, :, 1:]))
    return tv_loss


def consistency_loss_fn(original_output, transformed_output):
    loss = F.mse_loss(original_output, transformed_output)
    return loss


def semantic_consistency_loss_fn(original_output, transformed_output):
    cos_sim = F.cosine_similarity(original_output.view(1, -1), transformed_output.view(1, -1), dim=1)
    loss = 1 - cos_sim
    return loss.mean()


def entropy_loss_fn(output):
    epsilon = 1e-9
    entropy = -torch.sum(output * torch.log(output + epsilon))
    return entropy


def diversity_loss_fn(attn_maps):
    # Example: Pairwise diversity loss
    diversity = 0
    num_maps = len(attn_maps)
    if num_maps > 1:
        for i in range(num_maps):
            for j in range(i + 1, num_maps):
                diversity += F.mse_loss(attn_maps[i], attn_maps[j])
        diversity /= (num_maps * (num_maps - 1)) / 2
    return diversity


def semantic_diversity_loss_fn(attn_maps):
    diversity = 0
    num_maps = len(attn_maps)
    if num_maps > 1:
        for i in range(num_maps):
            for j in range(i + 1, num_maps):
                cos_sim = F.cosine_similarity(attn_maps[i].view(1, -1), attn_maps[j].view(1, -1), dim=1)
                diversity += (1 - cos_sim)
        diversity /= (num_maps * (num_maps - 1)) / 2
    return diversity


def optimize_embeddings(ldm, train_dataloader, val_dataloader,
                        context=None, num_tokens=100, device="cuda",
                        layers=[0, 1, 2, 3, 4, 5], noise_level=-1,
                        from_where=["down_cross", "mid_cross", "up_cross"], num_classes=10,
                        losses=None,
                        ):
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

    linear_layer = torch.nn.Sequential(
        # torch.nn.Dropout(0.2),
        # torch.nn.Linear(1000, 10),
        torch.nn.Linear(num_tokens, 10),
        # torch.nn.ReLU(),
        # torch.nn.Linear(256, 64),
        # torch.nn.ReLU(),
        # torch.nn.Linear(64, 10),
    ).to(device)

    # augment_degrees = 15
    # augment_scale = (0.9, 1.1)
    # augment_translate = (0.1, 0.1)
    augment_degrees = 35
    augment_scale = (0.8, 1.2)
    augment_translate = (0.2, 0.2)


    # linear_layer = torch.nn.Linear(num_tokens, num_classes).to(device)
    linear_layer.requires_grad = True

    # random project from 1000 to 10

    # optimizer = torch.optim.Adam([context], lr=1e-3)
    # optimizer = torch.optim.Adam([context, linear_layer.weight, linear_layer.bias], lr=5e-3)
    optimizer = torch.optim.Adam([
        {'params': context},
        {'params': linear_layer.parameters()}
    ], lr=5e-3)
    # ], lr=1e-3)
    cross_entropy = torch.nn.CrossEntropyLoss()

    # all traininalbe parameters

    dataloader_iter = iter(train_dataloader)

    equiloss = True if "equivariance_loss" in losses else False
    total_variation = True if "total_variation" in losses else False
    consistency = True if "consistency" in losses else False
    entropy_loss = True if "entropy_loss" in losses else False
    diversity_loss = True if "diversity_loss" in losses else False
    semantic_diversity_loss = True if "semantic_diversity_loss" in losses else False
    semantic_consistency_loss = True if "semantic_consistency_loss" in losses else False

    import torch.nn.utils as utils


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

            if equiloss or consistency or semantic_consistency_loss:
                invertible_transform = RandomAffineWithInverse(
                    degrees=augment_degrees,
                    scale=augment_scale,
                    translate=augment_translate,
                )

                transformed_img = invertible_transform(images)
                attention_maps_transformed = run_and_find_attn(
                    ldm,
                    transformed_img,
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

            cross_entropy_loss = cross_entropy(output, label)
            cross_entropy_loss /= 10000
            loss += cross_entropy_loss

            if equiloss:
                equi_loss = equivariance_loss(attn_maps[0], attention_maps_transformed[0][None].repeat(1, 1, 1, 1),
                                              invertible_transform, 0)
                equi_loss /= 100
                loss += equi_loss

            if total_variation:
                tv_loss = total_variation_loss_fn(attn_maps[0])
                loss += tv_loss

            if consistency:
                # original_output = attn_maps[0]
                # transformed_output = attention_maps_transformed[0]
                # cons_loss = consistency_loss_fn(original_output, transformed_output)
                # loss += cons_loss
                transformed_maps_means = torch.mean(attention_maps_transformed[0], dim=(1, 2))
                cos_sim = F.cosine_similarity(attn_map.view(1, -1), transformed_maps_means.view(1, -1), dim=1)
                cons_loss = 1 - cos_sim.mean()
                cons_loss /= 100
                loss += cons_loss

            if entropy_loss:
                ent_loss = entropy_loss_fn(output)
                ent_loss /= 10000
                loss += ent_loss

            if diversity_loss:
                div_loss = diversity_loss_fn(attn_maps[0])
                loss += div_loss

            if semantic_diversity_loss:
                sem_div_loss = semantic_diversity_loss_fn(attn_maps[0])
                sem_div_loss /= 100
                loss += sem_div_loss[0]

            if semantic_consistency_loss:
                sem_cons_loss = semantic_consistency_loss_fn(attn_maps[0], attention_maps_transformed[0])
                sem_cons_loss /= 100
                loss += sem_cons_loss



        loss /= len(labels)
        loss.backward()
        utils.clip_grad_norm_(linear_layer.parameters(), max_norm=1.0)
        # utils.clip_grad_norm_(context, max_norm=1.0)
        if (i + 1) % len(labels) == 0:
            optimizer.step()
            optimizer.zero_grad()

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
                    # if idx == 0:
                    #     plt.imshow(images[0].permute(1, 2, 0).detach().cpu())
                    #     plt.show()
                    #     plt.imshow(torch.mean(attn_map, dim=0).detach().cpu())
                    #     plt.show()
                    #     plt.imshow(attn_map[0].detach().cpu())
                    #     plt.show()
                    attn_maps = torch.mean(attn_map, dim=(1, 2))
                    outputs = linear_layer(attn_maps)

                    # attn_maps = attention_aggregator(attn_map.unsqueeze(0))
                    # outputs = torch.nn.functional.softmax(attn_maps, dim=1)
                    # outputs = outputs.squeeze(0)

                    predicted = torch.argmax(outputs, 0)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Epoch: {i}")
            # print(f"Loss: {loss.item()}")
            print(f"Accuracy: {100 * correct / total}")
            shuffled = torch.randperm(labels.size(0))
            print("ce loss", cross_entropy_loss.item())
            if equiloss:
                print("equi_loss", equi_loss.item())
            if total_variation:
                print("tv_loss", tv_loss.item())
            if entropy_loss:
                print("ent_loss", ent_loss.item())
            if consistency:
                print("cons_loss", cons_loss.item())
            if diversity_loss:
                print("div_loss", div_loss.item())
            if semantic_diversity_loss:
                print("sem_div_loss", sem_div_loss.item())
            if semantic_consistency_loss:
                print("sem_cons_loss", sem_cons_loss.item())
            # shuffled according to the labels

    return context.detach()


def parseargs():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Keypoints')
    parser.add_argument('--num_tokens', type=int, default=100)
    # options are: equivariance_loss, total_variation, consistency, entropy_loss, diversity_loss
    parser.add_argument('--losses', nargs='+',
                        default=["total_variation", "entropy_loss", "diversity_loss", "semantic_diversity_loss"])
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()
    trainloader, testloader = get_cifar10_dataloaders()

    ldm, controllers, num_gpus = load_ldm("cuda",
                                          "runwayml/stable-diffusion-v1-5",
                                          feature_upsample_res=128)

    context = optimize_embeddings(ldm, trainloader,
                                  val_dataloader=testloader, device="cuda",
                                  num_tokens=args.num_tokens, losses=args.losses)
