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

from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from einops import rearrange
from stable_diffusion.ldm.modules.diffusionmodules.util import timestep_embedding
import torch.nn.functional as F

from StableKeypoints import load_ldm, image2latent, init_random_noise, run_and_find_attn, RandomAffineWithInverse, \
    equivariance_loss

from datasets.thumos import ThumosClassificationDataset, ThumosDataset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


def get_thumos_dataloaders(batch_size=1, num_workers=4, resize_to=512, train_videos=None, test_videos=None):
    # transforms_train = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((resize_to, resize_to)),  # Resize the images
    #     transforms.ToTensor(),
    # ])
    #
    # transforms_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((resize_to, resize_to)),  # Resize the images
    #     transforms.ToTensor(),
    # ])

    train_set = ThumosClassificationDataset(split='validation', sampling_rate=8, n_videos=train_videos,
                                            transforms=None)
    test_set = ThumosClassificationDataset(split='test', sampling_rate=80, n_videos=test_videos,
                                           transforms=None)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def optimize_embeddings(ldm, train_dataloader, val_dataloader,
                        context=None, num_tokens=100, device="cuda",
                        layers=[0, 1, 2, 3, 4, 5], noise_level=-1,
                        from_where=["down_cross", "mid_cross", "up_cross"], num_classes=10,
                        losses=None,
                        extraction_method="return_maps",
                        ckpt=None
                        ):
    if context is None:
        context = init_random_noise(device, num_words=num_tokens)

    context.requires_grad = True

    linear_layer = torch.nn.Sequential(
        torch.nn.Linear(num_tokens, num_classes),
    ).to(device)

    linear_layer.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': context},
        {'params': linear_layer.parameters()},
    ], lr=0.005)

    freqs = torch.tensor(train_dataloader.dataset.get_freqs()).float()
    weight = torch.tensor(freqs.mean() / freqs).to(device)

    cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)

    dataloader_iter = iter(train_dataloader)

    results_dict = {}

    if extraction_method == "return_maps":
        config_path = "stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, f"{ckpt}")
        model = model.to(device)

    train_loss_hist = []
    test_loss_hist = []

    for i in tqdm(range(1_500_000)):
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

            if extraction_method == "hooks":
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

            elif extraction_method == "return_maps":
                _, attn_map = get_diff_latents(model,
                                               image.to(device),
                                               layers=[2, 5, 8, 12],
                                               t=100,
                                               resize=(512, 512),
                                               context=context)

            else:
                raise ValueError(f"Invalid extraction method: {extraction_method}")


            attn_map = torch.mean(attn_map, dim=(1, 2))
            output = linear_layer(attn_map)
            # output = torch.nn.functional.softmax(output, dim=0)

            cross_entropy_loss = cross_entropy(output, label)
            loss += cross_entropy_loss

            image.to('cpu')

        loss /= len(labels)
        loss.backward()
        # import torch.optim as optim

        train_loss_hist.append(loss.item())

        import torch.nn.utils as utils
        utils.clip_grad_norm_(linear_layer.parameters(), max_norm=1.0)
        utils.clip_grad_norm_(context, max_norm=1.0)

        if i % 1000 == 0:
            plot_gradients(context, extraction_method, i)

        if (i + 1) % len(labels) == 0:
            optimizer.step()
            optimizer.zero_grad()

        if not os.path.exists("./images"):
            os.makedirs("./images")
        if not os.path.exists("./attn_maps"):
            os.makedirs("./attn_maps")

        if i > 0 and i % 100_000 == 0:
        # if False:
            # validation
            correct = 0
            total = 0
            actions_confusion_matrix = torch.zeros(num_classes, num_classes)
            test_loss = 0

            with torch.no_grad():
                for idx, (images, labels) in enumerate(val_dataloader):
                    # labels = labels.to(device)
                    labels = labels.to(device).unsqueeze(0)

                    if extraction_method == "hooks":
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

                    elif extraction_method == "return_maps":
                        _, attn_map = get_diff_latents(model,
                                                       images.to(device),
                                                       layers=[2, 5, 8, 12],
                                                       t=100,
                                                       resize=(512, 512),
                                                       context=context)

                    else:
                        raise ValueError(f"Invalid extraction method: {extraction_method}")

                    if idx < 20:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(((images[0].permute(1, 2, 0).detach().cpu()) + 1) / 2)
                        # plt.show()
                        # plt.savefig(f"./images/{i}_0.png")
                        ax[1].imshow(attn_map[0].detach().cpu())
                        # plt.show()
                        # plt.savefig(f"./attn_maps/{i}_0.png")
                        os.makedirs(f"./exps/thumos/{extraction_method}/images_and_maps", exist_ok=True)
                        plt.savefig(f"./exps/thumos/{extraction_method}/images_and_maps/{i}_{idx}.png")

                    attn_maps = torch.mean(attn_map, dim=(1, 2))
                    outputs = linear_layer(attn_maps)

                    predicted = torch.argmax(outputs, 0)
                    # print(predicted, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # actions_confusion_matrix += confusion_matrix(labels, predicted,
                    #                                              labels=list(range(num_classes)))
                    actions_confusion_matrix[labels.item(), predicted.item()] += 1

                    test_loss += cross_entropy(outputs, labels.squeeze()).item()

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            disp = ConfusionMatrixDisplay(confusion_matrix=actions_confusion_matrix.numpy(),
                                          display_labels=ThumosDataset.categories + ["No action"])
            disp.plot(ax=ax, xticks_rotation='vertical')
            os.makedirs(f"./exps/thumos/{extraction_method}/confusion_matrix/disp", exist_ok=True)
            os.makedirs(f"./exps/thumos/{extraction_method}/confusion_matrix/npy", exist_ok=True)
            plt.savefig(f"./exps/thumos/{extraction_method}/confusion_matrix/disp/{i}.png")
            with open(f"./exps/thumos/{extraction_method}/confusion_matrix/npy/{i}.npy", "wb") as f:
                np.save(f, actions_confusion_matrix.numpy())

            acc = 100 * correct / total
            window_size = len(val_dataloader)
            train_loss = sum(train_loss_hist[-window_size:]) / window_size
            test_loss /= len(val_dataloader)
            test_loss_hist.append(test_loss)

            log_lines = (f"Step: {i} | Accuracy: {acc:.4}% "
                        f"| train loss: {train_loss:.4} | test loss: {test_loss:.4}\n")
            for idx, cls_name in enumerate(ThumosDataset.categories + ["No action"]):
                log_lines += (f" | {cls_name}: "
                              f"precision: {actions_confusion_matrix[idx, idx] / actions_confusion_matrix[:, idx].sum():.4} "
                              f"| recall: {actions_confusion_matrix[idx, idx] / actions_confusion_matrix[idx].sum():.4}\n")
            print(log_lines)
            os.makedirs(f"./exps/thumos/{extraction_method}", exist_ok=True)
            with open(f"./exps/thumos/{extraction_method}/log.txt", "a") as f:
                f.write(log_lines + "\n")

            os.makedirs(f"./exps/thumos/{extraction_method}/checkpoints", exist_ok=True)
            with open(f"./exps/thumos/{extraction_method}/checkpoints/context_{i}.pt", "wb") as f:
                torch.save(context, f)

            plot_loss(train_loss_hist, test_loss_hist, extraction_method)

        # # save results as txt
        # os.makedirs(f"results/thumos/{extraction_method}", exist_ok=True)
        # with open(f"results/thumos/{extraction_method}/results.txt", "wb") as f:
        #     pickle.dump(results_dict, f)

    return context.detach()


def plot_loss(train_loss_hist, test_loss_hist, extraction_method):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(train_loss_hist, label="Train Loss")
    ax.plot(test_loss_hist, label="Test Loss")
    ax.legend()
    plt.savefig(f"./exps/thumos/{extraction_method}/loss.png")


def get_diff_latents(model, init_image, layers=[2, 5, 8, 12], t=1, resize=None,
                     prompt="a photo of a person", context=None, resize_attn_maps=128,
                     plot_attn_maps=False):
    # plt.imshow(init_image[0].permute(1, 2, 0).detach().cpu())
    # plt.show()

    layer_idx = layers

    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    timesteps = torch.tensor([t])

    unet = model.model.diffusion_model

    x = init_latent

    x = torch.cat([x] * 2)
    timesteps = torch.cat([timesteps] * 2)

    if context is None:
        unconditional_conditioning = model.get_learned_conditioning([""])
        cond = model.get_learned_conditioning([prompt])
    else:
        unconditional_conditioning = context
        cond = context
    c_crossattn = [torch.cat([unconditional_conditioning, cond])]

    t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(x.device)
    emb = unet.time_embed(t_emb)

    context = torch.cat(c_crossattn, 1)
    # context = unconditional_conditioning

    token_ids = model.cond_stage_model.tokenizer([prompt], truncation=True, max_length=model.cond_stage_model.max_length,
                                                 return_length=True, return_overflowing_tokens=False,
                                                 padding="max_length", return_tensors="pt")
    length = int(token_ids['length'])
    token_ids = [int(token_id) for token_id in token_ids['input_ids'][0]]
    tokens = [model.cond_stage_model.tokenizer._convert_id_to_token(token_id) for token_id in token_ids]

    hs = []
    attn_maps = []

    h = x.type(unet.dtype)
    idx = 0
    modules = unet.input_blocks + [unet.middle_block]
    for module in modules:
        h, attn = module(h, emb, context, return_attn_maps=torch.tensor(1.0))
        hs.append(h)
        if idx in layer_idx:
            n_heads, n_patches, _ = attn.shape
            H, W = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
            attn = rearrange(attn, 'b (h w) c -> b c h w', h=H, w=W)

            attn = attn[n_heads//2:, :, :, :]

            if plot_attn_maps:
                plot_attn_maps(attn, tokens, length, suffix=f'_layer{idx}')

            attn_maps.append(attn)
        idx += 1

    # if isinstance(layer_idx, int):
    #     latents = hs[layer_idx]
    #     if resize is not None:
    #         latents = F.interpolate(latents, size=resize, mode='bilinear', align_corners=True)
    if isinstance(layer_idx, list):
        # latents = [F.interpolate(hs[idx], size=resize, mode='bilinear', align_corners=True)
        #            for idx in layer_idx]
        # latents = torch.cat(latents, dim=1)

        attn_maps = [F.interpolate(attn_map, size=resize_attn_maps,
                                   mode='bilinear', align_corners=True).unsqueeze(dim=1)
                     for attn_map in attn_maps]
        attn_maps = torch.cat(attn_maps, dim=1)
    else:
        raise ValueError(f"Invalid layers: {layer_idx}")

    return _, attn_maps.mean(dim=(0,1))


def plot_attn_maps(attn, tokens, length, suffix=''):
    n_heads, c, h, w = attn.shape
    fig, axs = plt.subplots(n_heads+1, length, figsize=(20, 10))
    for i in range(n_heads):
        for j in range(length):
            ax = axs[i, j]
            ax.imshow(attn[i, j].squeeze().detach().cpu().numpy())

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(tokens[j])
            if j == 0:
                ax.set_ylabel(f'Head {i}')
    for j in range(length):
        ax = axs[n_heads, j]
        ax.imshow(attn[:, j].mean(0).squeeze().detach().cpu().numpy())

        ax.set_xticks([])
        ax.set_yticks([])

        if j == 0:
            ax.set_ylabel('Mean')

    plt.tight_layout()
    plt.savefig(f'diff_attn_maps_layer{suffix}.png', dpi=300)
    plt.show()


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

    # plt.show()


def parseargs():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Keypoints')
    parser.add_argument('--num_tokens', type=int, default=100)
    # options are: equivariance_loss, total_variation, consistency, entropy_loss, diversity_loss
    parser.add_argument('--losses', nargs='+',
                        default=[""])
    parser.add_argument('--convolutional', action='store_true', default=False)
    parser.add_argument('--extraction_method', type=str, default="return_maps",
                        choices=("return_maps", "hooks"))
    parser.add_argument('--ckpt', type=str,
                        default='stable_diffusion/checkpoints/v1-5-pruned-emaonly.ckpt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()
    trainloader, testloader = get_thumos_dataloaders(train_videos=1, test_videos=1)

    if args.extraction_method == "hooks":
        ldm, controllers, num_gpus = load_ldm("cuda",
                                              "runwayml/stable-diffusion-v1-5",
                                              feature_upsample_res=128)
    elif args.extraction_method == "return_maps":
        ldm, controllers, num_gpus = None, None, 1

    print("Extraction Method:", args.extraction_method)

    context = optimize_embeddings(ldm, trainloader,
                                  val_dataloader=testloader, device="cuda",
                                  num_tokens=args.num_tokens, losses=args.losses,
                                  extraction_method=args.extraction_method,
                                  ckpt=args.ckpt, num_classes=len(ThumosDataset.categories)+1)
