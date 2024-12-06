import pickle
import time

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from PIL import Image as PILImage
import os

from torch import einsum
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.utils as utils
from tqdm import tqdm

from omegaconf import OmegaConf

from difftad import DiffTAD
from ldm.util import instantiate_from_config
from einops import rearrange, reduce

from stable_diffusion.ldm.modules.diffusionmodules.util import timestep_embedding
import torch.nn.functional as F

from StableKeypoints import load_ldm, image2latent, init_random_noise, run_and_find_attn, RandomAffineWithInverse, \
    equivariance_loss

from datasets.thumos import ThumosClassificationDataset, ThumosDataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from difftad import DiffTAD
from resnet_classifier import ResNetTAD

import pandas as pd


def get_cifar10_dataloaders(batch_size=1, num_workers=4, resize_to=512, testset_size=1000):
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader


def get_thumos_dataloaders(batch_size=1, num_workers=1, resize_to=512,
                           chunk_size=1, train_videos=None, test_videos=None):


    if test_videos is not None and test_videos >= 10:
        subset_prop = 0.1
    else:
        subset_prop = None

    train_set = ThumosClassificationDataset(split='validation', sampling_rate=8, n_videos=train_videos,
                                            chunk_size=chunk_size, transforms=None, normalize=False, img_size=resize_to)
    test_set = ThumosClassificationDataset(split='test', sampling_rate=8, subset_prop=subset_prop, n_videos=test_videos,
                                           chunk_size=chunk_size, transforms=None, normalize=False, img_size=resize_to)

    train_set_reduced, _ = torch.utils.data.random_split(train_set, [0.1, 0.9])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    reduced_train_loader = torch.utils.data.DataLoader(train_set_reduced, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader, reduced_train_loader


def get_freqs(data):
    if isinstance(data, ThumosClassificationDataset):
        return data.get_freqs()
    else:
        return get_freqs(data.dataset)


def optimize_embeddings(ldm, train_dataloader,
                        reduced_train_loader, val_dataloader,
                        context=None, num_tokens=100, device="cuda",
                        layers=[0, 1, 2, 3, 4, 5], noise_level=-1,
                        from_where=["down_cross", "mid_cross", "up_cross"], num_classes=21,
                        losses=None,
                        extraction_method="difftad",
                        ckpt=None,
                        n_steps=1_500_000,
                        val_interval=100_000,
                        batch_size=1,
                        chunk_size=1,
                        debug=False,
                        n_attn_maps_plots=10,
                        window_len=100,
                        n_image_samples=20,
                        backbone_ckpt=None,
                        ):
    train_freqs = torch.tensor(get_freqs(train_dataloader)).float()
    val_freqs = torch.tensor(get_freqs(val_dataloader)).float()
    red_train_freqs = torch.tensor(get_freqs(reduced_train_loader)).float()

    if debug:
        window_len = 5
        val_interval = 5

        train_freqs += 100
        val_freqs += 100
        red_train_freqs += 100

    train_weight = torch.tensor(train_freqs.mean() / train_freqs).to(device)
    cross_entropy_train = torch.nn.CrossEntropyLoss(weight=train_weight)

    val_weight = torch.tensor(val_freqs.mean() / val_freqs).to(device)
    cross_entropy_val = torch.nn.CrossEntropyLoss(weight=val_weight)

    red_train_weight = torch.tensor(red_train_freqs.mean() / red_train_freqs).to(device)
    cross_entropy_red_train = torch.nn.CrossEntropyLoss(weight=red_train_weight)

    dataloader_iter = iter(train_dataloader)

    label_names = ThumosDataset.categories + ["No action"]

    optimizer = None
    backbone = None

    linear_layer = torch.nn.Sequential(
        torch.nn.Linear(num_tokens, num_classes),
    ).to(device)
    linear_layer.requires_grad = True

    if extraction_method == "difftad":
        backbone = DiffTAD(config_path="stable_diffusion/configs/stable-diffusion/v1-inference.yaml",
                           ckpt="stable_diffusion/checkpoints/v1-5-pruned-emaonly.ckpt",
                           n_tokens=num_tokens,
                           num_frames=chunk_size,
                           context_ckpt=backbone_ckpt).to(device)
        optimizer = torch.optim.Adam([
            {'params': backbone.context},
            {'params': linear_layer.parameters()},
        ], lr=0.005)
    elif extraction_method == "resnet":
        backbone = ResNetTAD().to(device)
        optimizer = torch.optim.Adam([
            {'params': backbone.parameters(), 'lr': 0.0005},
            {'params': linear_layer.parameters(), 'lr': 0.005},
        ])
    else:
        raise ValueError(f"Invalid extraction method: {extraction_method}")

    train_loss_hist, train_loss_steps = [], []
    test_loss_hist, test_loss_steps = [], []
    red_train_loss_hist, red_train_loss_steps = [], []

    for step in tqdm(range(n_steps)):
        try:
            images, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            images, labels = next(dataloader_iter)

        images, labels = images.to(device), labels.to(device)
        backbone.train()

        loss = 0
        # for image, label in tqdm(zip(images, labels)):
        # for image, label in zip(images, labels):

        if extraction_method == "difftad":
            attn_maps = backbone(images)
            attn_maps = rearrange(attn_maps, 'b c t h w -> (b t) c h w')
            attn_maps = reduce(attn_maps, 'B c h w -> B c', 'mean')
            outputs = linear_layer(attn_maps)
        elif extraction_method == "resnet":
            latents = backbone(images)
            latents = rearrange(latents, 'b c t h w -> (b t) c h w')
            latents = reduce(latents, 'B c h w -> B c', 'mean')
            outputs = linear_layer(latents)
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")


        labels = rearrange(labels, 'b t -> (b t)')

        # output = torch.nn.functional.softmax(output, dim=0)

        cross_entropy_loss = cross_entropy_train(outputs, labels)
        loss += cross_entropy_loss

        images.to('cpu')

        loss /= len(labels)
        loss.backward()
        # import torch.optim as optim

        train_loss_hist.append(loss.item())
        train_loss_steps.append(step)

        utils.clip_grad_norm_(linear_layer.parameters(), max_norm=1.0)

        if extraction_method == "difftad":
            utils.clip_grad_norm_(backbone.context, max_norm=1.0)
        elif extraction_method == "resnet":
            utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")

        if step % 1000 == 0:
            if extraction_method == "difftad":
                plot_gradients(backbone.context, extraction_method, step)
            elif extraction_method == "resnet":
                pass
            else:
                raise ValueError(f"Invalid extraction method: {extraction_method}")

        optimizer.step()
        optimizer.zero_grad()

        if not os.path.exists("./images"):
            os.makedirs("./images")
        if not os.path.exists("./attn_maps"):
            os.makedirs("./attn_maps")

        if step > 0 and step % val_interval == 0:
            red_train_loss = validate("Reduced Train Set", reduced_train_loader,
                                      backbone, linear_layer, cross_entropy_val, label_names,
                                      extraction_method, step, train_loss_hist, num_classes,
                                      chunk_size, device, n_image_samples, n_attn_maps_plots)

            red_train_loss_hist.append(red_train_loss)
            red_train_loss_steps.append(step)

            test_loss = validate("Reduced Validation Set", val_dataloader,
                                 backbone, linear_layer, cross_entropy_red_train, label_names,
                                 extraction_method, step, train_loss_hist, num_classes,
                                 chunk_size, device, n_image_samples, n_attn_maps_plots)

            test_loss_hist.append(test_loss)
            test_loss_steps.append(step)

            os.makedirs(f"./exps/thumos/{extraction_method}/checkpoints", exist_ok=True)
            with open(f"./exps/thumos/{extraction_method}/checkpoints/context_{step}.pth", "wb") as f:
               if extraction_method == "difftad":
                   torch.save(backbone.context, f)
               elif extraction_method == "resnet":
                   torch.save(backbone.state_dict(), f)
               else:
                   raise ValueError(f"Invalid extraction method: {extraction_method}")


        plot_loss(train_loss_hist, train_loss_steps,
                  test_loss_hist, test_loss_steps,
                  red_train_loss_hist, red_train_loss_steps,
                  extraction_method, window_len)


def validate(dataset_name, val_dataloader, backbone, linear_layer, loss_func, label_names, extraction_method,
             step, train_loss_hist, num_classes, chunk_size, device,
             n_image_samples, n_attn_maps_plots):
    correct = 0
    total = 0
    actions_confusion_matrix = torch.zeros(num_classes, num_classes)
    test_loss = 0

    dataset_filename = dataset_name.replace(" ", "_").lower()

    with torch.no_grad():
        backbone.eval()
        for idx, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            if extraction_method == "difftad":
                attn_maps = backbone(images, return_attn_maps=True)

                images = rearrange(images, 'b c t h w -> b t c h w')
                attn_maps = rearrange(attn_maps, 'b c t h w -> b t c h w')

                if idx < n_image_samples:
                    fig, axes = plt.subplots(chunk_size, n_attn_maps_plots + 1,
                                             figsize=(n_attn_maps_plots, chunk_size))
                    for row in range(chunk_size):
                        img = images[0][row].detach().cpu()
                        attn = attn_maps[0][row].detach().cpu()
                        label = labels[0][row].detach().cpu()

                        ax = axes[row, 0]
                        ax.imshow(img.permute(1, 2, 0))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_ylabel(f"t={row}\n{label_names[label.item()]}")

                        for i in range(n_attn_maps_plots):
                            ax = axes[row, i + 1]
                            ax.imshow(attn[i])
                            ax.set_xticks([])
                            ax.set_yticks([])

                    os.makedirs(f"./exps/thumos/{extraction_method}/{dataset_filename}/images_and_maps", exist_ok=True)
                    plt.savefig(f"./exps/thumos/{extraction_method}/{dataset_filename}//images_and_maps{step}_{idx}.png", dpi=300)
                    plt.close()
                    # plt.show()

                    attn_maps = reduce(attn_maps, 'b t c h w -> (b t) c', 'mean')
                    outputs = linear_layer(attn_maps)

            elif extraction_method == "resnet":
                latents = backbone(images)
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = reduce(latents, 'B c h w -> B c', 'mean')
                outputs = linear_layer(latents)
            else:
                raise ValueError(f"Invalid extraction method: {extraction_method}")

            labels = rearrange(labels, 'b t -> (b t)')
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, pred in zip(labels, predicted):
                actions_confusion_matrix[label.item(), pred.item()] += 1

            test_loss += loss_func(outputs, labels).item()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=actions_confusion_matrix.numpy(),
                                  display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation='vertical')
    os.makedirs(f"./exps/thumos/{extraction_method}/{dataset_filename}/confusion_matrix/disp", exist_ok=True)
    os.makedirs(f"./exps/thumos/{extraction_method}/{dataset_filename}/confusion_matrix/npy", exist_ok=True)
    plt.savefig(f"./exps/thumos/{extraction_method}/{dataset_filename}/confusion_matrix/disp/{step}.png")
    plt.close()

    with open(f"./exps/thumos/{extraction_method}/{dataset_filename}/confusion_matrix/npy/{step}.npy", "wb") as f:
        np.save(f, actions_confusion_matrix.numpy())

    acc = 100 * correct / total
    window_size = len(val_dataloader)
    train_loss = sum(train_loss_hist[-window_size:]) / window_size
    test_loss /= len(val_dataloader)

    log_lines = (f"\n"
                 f"Testing on {dataset_name}\n"
                 f"Step: {step} | Accuracy: {acc:.4}% "
                 f"| train loss: {train_loss:.4} | test loss: {test_loss:.4}\n")
    for idx, cls_name in enumerate(label_names):
        log_lines += (f" | {cls_name}: "
                      f"precision: {actions_confusion_matrix[idx, idx] / actions_confusion_matrix[:, idx].sum():.4} "
                      f"| recall: {actions_confusion_matrix[idx, idx] / actions_confusion_matrix[idx].sum():.4}\n")
    print(log_lines)
    os.makedirs(f"./exps/thumos/{extraction_method}/{dataset_filename}", exist_ok=True)
    with open(f"./exps/thumos/{extraction_method}/{dataset_filename}/log.txt", "a") as f:
        f.write(log_lines + "\n")

    return test_loss


def plot_loss(train_loss_hist, train_loss_steps,
              test_loss_hist, test_loss_steps,
                red_train_loss_hist, red_train_loss_steps,
              extraction_method, window_len):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(train_loss_steps, train_loss_hist, color='blue', alpha=0.1)

    ax.plot(red_train_loss_steps, red_train_loss_hist, color='green', label="Reduced Train Loss")

    train_loss_df = pd.DataFrame({'Step': train_loss_steps, 'Loss': train_loss_hist})
    train_loss_df['Rolling Mean'] = train_loss_df['Loss'].rolling(window=window_len).mean()
    ax.plot(train_loss_steps, train_loss_df['Rolling Mean'], color='blue', label="Train Loss")

    ax.plot(test_loss_steps, test_loss_hist, color='orange', label="Test Loss")
    ax.legend()

    os.makedirs(f"./exps/thumos/{extraction_method}", exist_ok=True)
    plt.savefig(f"./exps/thumos/{extraction_method}/loss.png")
    plt.close()


def plot_gradients(context, extraction_method, step):
    # Get the gradients from the context tensor
    gradients = context.grad

    # Compute the absolute values of the gradients
    gradients_abs = torch.abs(gradients)

    # Convert the gradients to a NumPy array for plotting
    gradients_abs_np = gradients_abs.detach().cpu().numpy()

    gradients_abs_flat = gradients_abs_np.flatten()

    # Plot the histogram of the absolute gradient values
    # plt.hist(gradients_abs_flat, bins=50)

    # Set up logarithmic bins for the histogram
    # Define min and max values for the bins based on the data
    min_grad = max(np.min(gradients_abs_flat), 1e-10)  # avoid zero issues
    max_grad = np.max(gradients_abs_flat)

    # Generate logarithmically spaced bins
    bins = np.logspace(np.log10(min_grad), np.log10(max_grad), 100)

    # Plot the histogram of the absolute gradient values with log scale
    fig, ax = plt.subplots()
    plt.hist(gradients_abs_flat, bins=bins)

    # Set the x and y axes to log scale
    plt.xscale('log')
    plt.yscale('log')

    plt.title(f'Histogram of Absolute Gradient Values\n{extraction_method} - {step} steps\n'
              f' mean: {gradients_abs_flat.mean()}, std: {gradients_abs_flat.std()}')
    plt.xlabel('Absolute Gradient Value')
    plt.ylabel('Frequency')

    os.makedirs(f"./exps/thumos/{extraction_method}/gradients", exist_ok=True)
    plt.savefig(f"./exps/thumos/{extraction_method}/gradients/{step}.png", dpi=300)


def parseargs():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Keypoints')
    parser.add_argument('--num_tokens', type=int, default=100)
    # options are: equivariance_loss, total_variation, consistency, entropy_loss, diversity_loss
    parser.add_argument('--losses', nargs='+',
                        default=[""])
    parser.add_argument('--convolutional', action='store_true', default=False)
    parser.add_argument('--extraction_method', type=str, default="difftad",
                        choices=("difftad", "resnet"))
    parser.add_argument('--ckpt', type=str,
                        default='stable_diffusion/checkpoints/v1-5-pruned-emaonly.ckpt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=2)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--backbone_ckpt', type=str, default=None)
    parser.add_argument('--val_interval', type=int, default=100_000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()

    train_videos, test_videos = [1, 1] if args.debug else [None, None]
    trainloader, testloader, reducedTrainLoader = get_thumos_dataloaders(batch_size=args.batch_size, chunk_size=args.chunk_size,
                                                     train_videos=train_videos, test_videos=test_videos)

    if args.extraction_method != "hooks":
        ldm, controllers, num_gpus = None, None, 1

    print("Extraction Method:", args.extraction_method)

    context = optimize_embeddings(ldm, trainloader,
                                  val_dataloader=testloader, device="cuda",
                                  reduced_train_loader=reducedTrainLoader,
                                  num_tokens=args.num_tokens, losses=args.losses,
                                  extraction_method=args.extraction_method,
                                  ckpt=args.ckpt, num_classes=len(ThumosDataset.categories)+1,
                                  batch_size=args.batch_size,
                                  chunk_size=args.chunk_size,
                                  # backbone_ckpt=args.backbone_ckpt,
                                  backbone_ckpt=args.backbone_ckpt,
                                  val_interval=args.val_interval,
                                  debug=args.debug)
