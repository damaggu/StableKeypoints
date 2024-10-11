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
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),  # Resize the images
        transforms.ToTensor(),
    ])

    dl = True if not os.path.exists('./data/imagenette2') else False

    trainset = torchvision.datasets.Imagenette(root='./data', download=dl, transform=transform_train, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.Imagenette(root='./data', download=dl, transform=transform_test, split='val')
    testset, _ = torch.utils.data.random_split(testset, [testset_size, len(testset) - testset_size])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def optimize_embeddings(ldm, train_dataloader, val_dataloader,
                        context=None, num_tokens=100, device="cuda",
                        layers=[0, 1, 2, 3, 4, 5], noise_level=-1,
                        from_where=["down_cross", "mid_cross", "up_cross"], num_classes=10,
                        losses=None,
                        ):
    if context is None:
        context = init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    linear_layer = torch.nn.Sequential(
        torch.nn.Linear(num_tokens, 10),
    ).to(device)

    linear_layer.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': context},
        {'params': linear_layer.parameters()},
    ], lr=0.005)

    cross_entropy = torch.nn.CrossEntropyLoss()

    dataloader_iter = iter(train_dataloader)

    results_dict = {}

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

            cross_entropy_loss = cross_entropy(output, label)
            loss += cross_entropy_loss

        loss /= len(labels)
        loss.backward()
        # import torch.optim as optim
        import torch.nn.utils as utils
        utils.clip_grad_norm_(linear_layer.parameters(), max_norm=1.0)
        utils.clip_grad_norm_(context, max_norm=1.0)
        if (i + 1) % len(labels) == 0:
            optimizer.step()
            optimizer.zero_grad()

        if not os.path.exists("./images"):
            os.makedirs("./images")
        if not os.path.exists("./attn_maps"):
            os.makedirs("./attn_maps")

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
                        # plt.show()
                        plt.savefig(f"./images/{i}_0.png")
                        plt.imshow(attn_map[0].detach().cpu())
                        # plt.show()
                        plt.savefig(f"./attn_maps/{i}_0.png")

                    attn_maps = torch.mean(attn_map, dim=(1, 2))
                    outputs = linear_layer(attn_maps)

                    predicted = torch.argmax(outputs, 0)
                    print(predicted, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Epoch: {i}")
            # print(f"Loss: {loss.item()}")
            print(f"Accuracy: {100 * correct / total}")
            shuffled = torch.randperm(labels.size(0))
            print("ce loss", cross_entropy_loss.item())
            results_dict[i] = {
                "loss": loss.item(),
                "accuracy": 100 * correct / total,
                "ce_loss": cross_entropy_loss.item(),
            }

        # save results as txt
        with open("results.txt", "wb") as f:
            pickle.dump(results_dict, f)

    return context.detach()


def parseargs():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Keypoints')
    parser.add_argument('--num_tokens', type=int, default=500)
    # options are: equivariance_loss, total_variation, consistency, entropy_loss, diversity_loss
    parser.add_argument('--losses', nargs='+',
                        default=[""])
    parser.add_argument('--convolutional', action='store_true', default=False)
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
