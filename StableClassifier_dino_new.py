# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy
import random
from typing import Any, Optional, List

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import torchvision
from lightly.data import LightlyDataset
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import os

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.ptp_utils import init_random_noise, run_and_find_attn

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

pl.seed_everything(42)

image_size = 224
max_epochs = 100
num_classes = 100
batch_size = 16
gradient_accumulation_steps = 16


def knn_predict(
        feature: Tensor,
        feature_bank: Tensor,
        feature_labels: Tensor,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
) -> Tensor:
    """Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN predictions

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback

    At the end of every training epoch we create a feature bank by feeding the
    `dataloader_kNN` passed to the module through the backbone.
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the
    feature_bank features from the train data.

    We can access the highest test accuracy during a kNN prediction
    using the `max_accuracy` attribute.

    Attributes:
        backbone:
            The backbone model used for kNN validation. Make sure that you set the
            backbone when inheriting from `BenchmarkModule`.
        max_accuracy:
            Floating point number between 0.0 and 1.0 representing the maximum
            test accuracy the benchmarked model has achieved.
        dataloader_kNN:
            Dataloader to be used after each training epoch to create feature bank.
        num_classes:
            Number of classes. E.g. for cifar10 we have 10 classes. (default: 10)
        knn_k:
            Number of nearest neighbors for kNN
        knn_t:
            Temperature parameter for kNN

    Examples:
        >>> class SimSiamModel(BenchmarkingModule):
        >>>     def __init__(dataloader_kNN, num_classes):
        >>>         super().__init__(dataloader_kNN, num_classes)
        >>>         resnet = lightly.models.ResNetGenerator('resnet-18')
        >>>         self.backbone = nn.Sequential(
        >>>             *list(resnet.children())[:-1],
        >>>             nn.AdaptiveAvgPool2d(1),
        >>>         )
        >>>         self.resnet_simsiam =
        >>>             lightly.models.SimSiam(self.backbone, num_ftrs=512)
        >>>         self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        >>>
        >>>     def forward(self, x):
        >>>         self.resnet_simsiam(x)
        >>>
        >>>     def training_step(self, batch, batch_idx):
        >>>         (x0, x1), _, _ = batch
        >>>         x0, x1 = self.resnet_simsiam(x0, x1)
        >>>         loss = self.criterion(x0, x1)
        >>>         return loss
        >>>     def configure_optimizers(self):
        >>>         optim = torch.optim.SGD(
        >>>             self.resnet_simsiam.parameters(), lr=6e-2, momentum=0.9
        >>>         )
        >>>         return [optim]
        >>>
        >>> model = SimSiamModel(dataloader_train_kNN)
        >>> trainer = pl.Trainer()
        >>> trainer.fit(
        >>>     model,
        >>>     train_dataloader=dataloader_train_ssl,
        >>>     val_dataloaders=dataloader_test
        >>> )
        >>> # you can get the peak accuracy using
        >>> print(model.max_accuracy)

    """

    def __init__(
            self,
            dataloader_kNN: DataLoader,
            num_classes: int,
            knn_k: int = 10,
            knn_t: float = 0.1,
    ):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t

        self._train_features: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None
        self._val_predicted_labels: List[Tensor] = []
        self._val_targets: List[Tensor] = []

    def on_validation_epoch_start(self) -> None:
        train_features = []
        train_targets = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                # TODO: changes here
                img = img[0]
                target = target[0]
                img = img.to(self.device)
                target = target.to(self.device).unsqueeze(0)
                if "Stable" in self.__class__.__name__:
                    feature = self.get_attn_maps(img, self.student_backbone)
                    feature = feature.unsqueeze(0)
                else:
                    feature = self(img)
                feature = F.normalize(feature, dim=1)
                if (
                        dist.is_available()
                        and dist.is_initialized()
                        and dist.get_world_size() > 0
                ):
                    # gather features and targets from all processes
                    feature = torch.cat(dist.gather(feature), 0)
                    target = torch.cat(dist.gather(target), 0)
                train_features.append(feature)
                train_targets.append(target)
        self._train_features = torch.cat(train_features, dim=0).t().contiguous()
        self._train_targets = torch.cat(train_targets, dim=0).t().contiguous()

    def validation_step(self, batch, batch_idx) -> None:
        # we can only do kNN predictions once we have a feature bank
        if self._train_features is not None and self._train_targets is not None:
            images, targets, _ = batch
            if "Stable" in self.__class__.__name__:
                feature = self.get_attn_maps(images[0], self.student_backbone)
                feature = feature.unsqueeze(0)
            else:
                feature = self(images[0])
            feature = F.normalize(feature, dim=1)
            predicted_labels = knn_predict(
                feature,
                self._train_features,
                self._train_targets,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )
            if dist.is_initialized() and dist.get_world_size() > 0:
                # gather predictions and targets from all processes
                predicted_labels = torch.cat(dist.gather(predicted_labels), 0)
                targets = torch.cat(dist.gather(targets), 0)

            self._val_predicted_labels.append(predicted_labels.cpu())
            self._val_targets.append(targets.cpu())

    def on_validation_epoch_end(self) -> None:
        if self._val_predicted_labels and self._val_targets:
            predicted_labels = torch.cat(self._val_predicted_labels, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            top1 = (predicted_labels[:, 0] == targets).float().sum()
            acc = top1 / len(targets)
            if acc > self.max_accuracy:
                self.max_accuracy = acc.item()
            self.log("kNN_accuracy", acc * 100.0, prog_bar=True)
            print(f"Accuracy: {acc * 100.0:.2f}%")

        self._val_predicted_labels.clear()
        self._val_targets.clear()


class DINO(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        hidden_dim = 512
        bottleneck_dim = 64
        output_dim = 2048
        freeze_last_layer = 1

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=freeze_last_layer
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, hidden_dim, bottleneck_dim, output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        if batch_idx % 100 == 0:
            l = loss.detach().cpu().item()
            self.log("train_loss", l)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


import matplotlib.pyplot as plt

ldm, controllers, num_gpus = load_ldm("cuda:0",
                                      "runwayml/stable-diffusion-v1-5",
                                      feature_upsample_res=128)


def update_momentum_tensor(tensor: torch.Tensor, tensor_ema: torch.Tensor, m: float):
    """
    Updates elements of `tensor_ema` with Exponential Moving Average of `tensor`.

    Momentum encoders are a crucial component of models such as MoCo or BYOL.

    Examples:
        >>> tensor = torch.randn(10, 10)
        >>> tensor_ema = torch.randn(10, 10)
        >>> update_momentum_tensor(tensor, tensor_ema, m=0.999)
    """
    if tensor.shape != tensor_ema.shape:
        raise ValueError("Input tensors must have the same shape")

    tensor_ema.data = tensor_ema.data * m + tensor.data * (1.0 - m)


transform = DINOTransform(normalize=None, global_crop_size=image_size)

try:
    dataset = torchvision.datasets.FGVCAircraft(
        "datasets/aircraft",
        # "datasets/imagenette",
        download=True,
        transform=transform,
    )
except:
    dataset = torchvision.datasets.FGVCAircraft(
        "datasets/aircraft",
        # "datasets/imagenette",
        download=False,
        transform=transform,
    )

print(len(dataset))
plt.imshow(dataset[0][0][0].permute(1, 2, 0))
plt.show()

# train / test split
dataset_train = torch.utils.data.Subset(dataset, range(0, len(dataset), 2))
dataset_test = torch.utils.data.Subset(dataset, range(1, len(dataset), 2))

example_frame = dataset[0][0][0]


class StableDINO(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        layers = [0, 1, 2, 3, 4, 5]
        noise_level = -1
        from_where = ["down_cross", "mid_cross", "up_cross"]
        self.layers = layers
        self.noise_level = noise_level
        self.from_where = from_where

        # hidden_dim = 512
        # bottleneck_dim = 64
        # output_dim = 2048

        hidden_dim = 256
        bottleneck_dim = 32
        output_dim = 1024

        num_tokens = 500
        backbone = init_random_noise(self.device, num_words=num_tokens)
        input_dim = num_tokens

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, hidden_dim, bottleneck_dim, output_dim)

        self.student_backbone.requires_grad = True
        self.student_head.requires_grad = True
        self.teacher_backbone.requires_grad = False
        self.teacher_head.requires_grad = False

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=5)

    def get_attn_maps(self, x, backbone=None):
        attn_maps = run_and_find_attn(
            ldm,
            x,
            backbone,
            layers=self.layers,
            noise_level=self.noise_level,
            from_where=self.from_where,
            upsample_res=-1,
            device=self.device,
            controllers=controllers,
        )
        attn_maps = attn_maps[0]
        return torch.mean(attn_maps, dim=(1, 2))

    def get_attention_visualization(self, x, backbone=None):
        attn_maps = run_and_find_attn(
            ldm,
            x,
            backbone,
            layers=self.layers,
            noise_level=self.noise_level,
            from_where=self.from_where,
            upsample_res=-1,
            device=self.device,
            controllers=controllers,
        )
        attn_maps = attn_maps[0]
        return attn_maps

    def forward(self, x):
        # y = self.student_backbone(x).flatten(start_dim=1)
        attn_maps = [self.get_attn_maps(i.unsqueeze(0), self.student_backbone) for i in x]
        y = torch.stack(attn_maps)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        attn_maps = [self.get_attn_maps(i.unsqueeze(0), self.teacher_backbone) for i in x]
        y = torch.stack(attn_maps)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum_tensor(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        # print("student weights:")
        # print(self.student_head.layers[0].weight[0][0])
        # print("student backbones:")
        # print(self.student_backbone[0][0][0])

        # if batch_idx % 100 == 0:
        if batch_idx % 10 == 0:
            resfolder = "results_dino"
            if not os.path.exists(resfolder):
                os.makedirs(resfolder)
            print(f"Epoch {self.current_epoch}, Loss {loss}")
            # show the image
            plt.imshow(example_frame.permute(1, 2, 0).detach().cpu())
            # plt.show()
            plt.savefig(os.path.join(resfolder, f"example_frame_{self.current_epoch}_{batch_idx}.png"))
            # show the attention map
            with torch.no_grad():
                attn_map = self.get_attention_visualization(example_frame.unsqueeze(0), self.student_backbone)
                plt.imshow(attn_map[0].detach().cpu())
                # plt.show()
                # save the maps
                plt.savefig(os.path.join(resfolder, f"attn_map0_{self.current_epoch}_{batch_idx}.png"))
                plt.imshow(attn_map[1].detach().cpu())
                # plt.show()
                # save the maps
                plt.savefig(os.path.join(resfolder, f"attn_map1_{self.current_epoch}_{batch_idx}.png"))
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        # optim = torch.optim.Adam(self.parameters(), lr=0.001)
        optim = torch.optim.Adam([
            {'params': self.student_backbone},
            {'params': self.student_head.parameters()},
        ], lr=0.001)

        return optim


dataloader_train = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
    ]
)

dataset_test_kNN = LightlyDataset.from_torch_dataset(dataset_test, transform=test_transforms)
# subset the dataset to only use a fraction of the data - only length 20 random samples
dataset_test_kNN = torch.utils.data.Subset(dataset_test_kNN,
                                           random.sample(range(len(dataset_test_kNN)), 20))

dataloader_val = torch.utils.data.DataLoader(
    dataset_test_kNN,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)

# model = StableDINO(dataloader_val, num_classes)
model = DINO(dataloader_val, num_classes)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(max_epochs=max_epochs, devices=8, accelerator=accelerator, accumulate_grad_batches=gradient_accumulation_steps)
trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
