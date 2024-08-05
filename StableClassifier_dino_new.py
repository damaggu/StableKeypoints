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
from lightly.transforms import MoCoV2Transform
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from lightly.loss import DINOLoss, NTXentLoss
from lightly.models.modules import DINOProjectionHead, MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from tqdm import tqdm

from unsupervised_keypoints.optimize_token import load_ldm
from unsupervised_keypoints.ptp_utils import init_random_noise, run_and_find_attn

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

pl.seed_everything(42)

image_size = 256
max_epochs = 500
num_classes = 100
# batch_size = 8
batch_size = 2
debug = 0


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
                # img = img.to(self.device)
                # target = target.to(self.device)

                # # TODO: changes here
                img = img[0]
                target = target[0]
                img = img.to(self.device)
                target = target.to(self.device).unsqueeze(0)

                # feature = self.get_attn_maps(img, self.student_backbone)
                # features = [self.get_attn_maps(i.unsqueeze(0), self.student_backbone) for i in img]
                # feature = torch.stack(features)

                if 'Stable' in self.__class__.__name__:
                    feature = self.forward(img)
                else:
                    feature = self.backbone(img)
                    feature = feature[:, :, -1, -1]  # 128, 512, 1, 1-> 128, 512
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
            # feature = self.get_attn_maps(images[0], self.student_backbone)
            # feature = feature.unsqueeze(0)
            images = images[0]
            images = images.to(self.device)
            targets = targets.to(self.device)
            # feature = self.backbone(images).squeeze()

            if 'Stable' in self.__class__.__name__:
                feature = self.forward(images)
            else:
                feature = self.backbone(images)
                feature = feature[:, :, -1, -1]  # 128, 512, 1, 1-> 128, 512
            feature = F.normalize(feature, dim=1)

            # features = [self.get_attn_maps(i.unsqueeze(0), self.student_backbone) for i in images]
            # feature = torch.stack(features)
            # feature = F.normalize(feature, dim=1)

            # feature = F.normalize(feature, dim=1)
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

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

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
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


import matplotlib.pyplot as plt

ldm, controllers, num_gpus = load_ldm("cuda:0",
                                      "runwayml/stable-diffusion-v1-5",
                                      feature_upsample_res=32)


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


# transform = DINOTransform(normalize=None, global_crop_size=image_size)
transform = MoCoV2Transform(input_size=image_size, normalize=None)

try:
    dataset = torchvision.datasets.Imagenette(
        # "datasets/aircraft",
        "datasets/imagenette",
        download=True,
        transform=transform,
    )
except:
    dataset = torchvision.datasets.Imagenette(
        # "datasets/aircraft",
        "datasets/imagenette",
        download=False,
        transform=transform,
    )

print(len(dataset))
# plt.imshow(dataset[0][0][0].permute(1, 2, 0))
# plt.show()

# train / test split
dataset_train = torch.utils.data.Subset(dataset, range(0, len(dataset), 2))
dataset_test = torch.utils.data.Subset(dataset, range(1, len(dataset), 2))

# debug
if debug:
    dataset_train = torch.utils.data.Subset(dataset_train, range(0, 50))
    dataset_test = torch.utils.data.Subset(dataset_test, range(0, 50))

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

        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch}, Loss {loss}")
            # show the image
            plt.imshow(example_frame.permute(1, 2, 0).detach().cpu())
            # plt.show()
            plt.savefig(f"example_frame_{self.current_epoch}_{batch_idx}.png")
            # show the attention map
            with torch.no_grad():
                attn_map = self.get_attention_visualization(example_frame.unsqueeze(0), self.student_backbone)
                plt.imshow(attn_map[0].detach().cpu())
                # plt.show()
                # save the maps
                plt.savefig(f"attn_map0_{self.current_epoch}_{batch_idx}.png")
                plt.imshow(attn_map[1].detach().cpu())
                # plt.show()
                # save the maps
                plt.savefig(f"attn_map1_{self.current_epoch}_{batch_idx}.png")
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


class MoCo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.01)
        return optim


class StableMoCo(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        layers = [0, 1, 2, 3, 4, 5]
        noise_level = -1
        from_where = ["down_cross", "mid_cross", "up_cross"]
        self.layers = layers
        self.noise_level = noise_level
        self.from_where = from_where

        hidden_dim = 256
        bottleneck_dim = 32
        output_dim = 1024

        num_tokens = 1000
        # make student backbone parameter
        self.student_backbone = nn.Parameter(torch.randn(1, num_tokens, 768))
        self.student_backbone.requires_grad = True
        input_dim = num_tokens

        # self.projection_head = MoCoProjectionHead(input_dim, input_dim, 128)
        # self.projection_head = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     # nn.ReLU(),
        #     # nn.Linear(input_dim, 128),
        # )
        self.projection_head = nn.Linear(input_dim, 128)
        self.projection_head.requires_grad = True

        self.backbone_momentum = copy.deepcopy(self.student_backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.backbone_momentum.requires_grad = False
        deactivate_requires_grad(self.projection_head_momentum)

        for p in self.named_parameters():
            if p[1].requires_grad:
                print(p[0])

        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

    def get_attn_maps(self, x, bbone=None):
        attn_maps = run_and_find_attn(
            ldm,
            x,
            bbone,
            layers=self.layers,
            noise_level=self.noise_level,
            from_where=self.from_where,
            upsample_res=-1,
            device=self.device,
            controllers=controllers,
        )
        attn_maps = attn_maps[0]
        return torch.mean(attn_maps, dim=(1, 2))

    def get_attention_visualization(self, x, bbone=None):
        attn_maps = run_and_find_attn(
            ldm,
            x,
            bbone,
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
        attn_maps = [self.get_attn_maps(i.unsqueeze(0), self.student_backbone) for i in x]
        attn_maps = torch.stack(attn_maps)
        query = self.projection_head(attn_maps)
        return query

    def forward_momentum(self, x):
        attn_maps = [self.get_attn_maps(i.unsqueeze(0), self.backbone_momentum) for i in x]
        attn_maps = torch.stack(attn_maps)
        key = self.projection_head_momentum(attn_maps).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum_tensor(self.student_backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        # loss.backward()
        # print(self.student_backbone.grad)
        print(self.student_backbone[0, 0, 0])
        # print(self.projection_head[0].weight.grad)
        print(self.projection_head[0].weight[0][1])

        if batch_idx % 100 == 0:
            print(f"Epoch {self.current_epoch}, Loss {loss}")
            # show the image
            plt.imshow(example_frame.permute(1, 2, 0).detach().cpu())
            # TODO: check if normalization is correct?
            plt.savefig(f"example_frame_{self.current_epoch}_{batch_idx}.png")
            # show the attention map
            with torch.no_grad():
                attn_map = self.get_attention_visualization(example_frame.unsqueeze(0), self.student_backbone)
                plt.imshow(attn_map[0].detach().cpu())
                plt.savefig(f"attn_map0_{self.current_epoch}_{batch_idx}.png")
                plt.imshow(attn_map[1].detach().cpu())
                plt.savefig(f"attn_map1_{self.current_epoch}_{batch_idx}.png")
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.SGD(self.parameters(), lr=0.06)
        optim = torch.optim.SGD([
            # {'params': self.student_backbone},
            {'params': self.parameters()},
            # {'params': self.projection_head.parameters()}
        ], lr=0.1)
        # optimizer = torch.optim.Adam([
        #     {'params': self.parameters(), 'weight_decay': 0.0001},
        #     {'params': self.student_backbone, 'weight_decay': 0.0}
        # ], lr=0.005)
        # return optimizer
        return optim


class NormalMoCo(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, 128))

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


if __name__ == "__main__":

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
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
    # model = StableMoCo(dataloader_val, num_classes)
    # model = NormalMoCo(dataloader_val, num_classes)
    # mdl = "normal_moco"
    mdl = "awgwg"

    if mdl == "normal_moco":
        model = NormalMoCo(dataloader_val, num_classes)
    else:
        model = StableMoCo(dataloader_val, num_classes)

    accelerator = "gpu"

    # trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator=accelerator, detect_anomaly=True)
    # trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    if mdl == "normal_moco":
        optimizer = torch.optim.SGD(
            [
                # {"params": model.student_backbone},
                {"params": model.backbone.parameters()},
                {"params": model.projection_head.parameters()},
            ],
            lr=0.06,
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.student_backbone},
                {"params": model.projection_head.parameters()},
            ],
            lr=0.06,
        )
    device = "cuda"
    model.to(device)
    model.train()
    criterion = NTXentLoss(memory_bank_size=(4096, 128))

    # print dataset len
    print(len(dataset))
    losses = []
    accs = []

    model.eval()
    with torch.no_grad():
        model.on_validation_epoch_start()
        for val_batch_idx, val_batch in enumerate(dataloader_val):
            model.validation_step(val_batch, val_batch_idx)
        model.on_validation_epoch_end()

    for epoch in range(max_epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, max_epochs, 0.996, 1)
        model.train()
        for batch_idx, batch in tqdm(enumerate(dataloader_train)):
            if mdl == "normal_moco":
                x_query, x_key = batch[0]
                update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
                update_momentum(
                    model.projection_head, model.projection_head_momentum, m=momentum_val
                )
                x_query = x_query.to(device)
                x_key = x_key.to(device)
                query = model(x_query)
                key = model.forward_momentum(x_key)
                loss = criterion(query, key)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                x_query, x_key = batch[0]
                update_momentum_tensor(model.student_backbone, model.backbone_momentum, m=momentum_val)
                update_momentum(
                    model.projection_head, model.projection_head_momentum, m=momentum_val
                )
                x_query = x_query.to(device)
                x_key = x_key.to(device)

                # model.student_backbone.retain_grad()
                # model.projection_head.retain_grad()
                query = model(x_query)
                key = model.forward_momentum(x_key)
                loss = criterion(query, key)
                total_loss += loss.detach()

                query.retain_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(model.student_backbone[0, 0, 0])
                print(model.projection_head.weight[0][1])
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Loss {loss}")
                    # show the image
                    plt.imshow(example_frame.permute(1, 2, 0).detach().cpu())
                    # plt.show()
                    plt.savefig(f"example_frame_{epoch}_{batch_idx}.png")
                    # show the attention map
                    with torch.no_grad():
                        attn_map = model.get_attention_visualization(example_frame.unsqueeze(0), model.student_backbone)
                        plt.imshow(attn_map[0].detach().cpu())
                        # plt.show()
                        plt.savefig(f"attn_map0_{epoch}_{batch_idx}.png")
                        plt.imshow(attn_map[1].detach().cpu())
                        # plt.show()
                        plt.savefig(f"attn_map1_{epoch}_{batch_idx}.png")
        model.eval()
        with torch.no_grad():
            print(model.backbone[0].weight[0][0][0][0])
            model.on_validation_epoch_start()
            for val_batch_idx, val_batch in enumerate(dataloader_val):
                model.validation_step(val_batch, val_batch_idx)
            model.on_validation_epoch_end()

        avg_loss = total_loss / len(dataloader_train)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        losses.append(avg_loss)
        accs.append(model.max_accuracy)

    # save the stats
    with open(f"stats_{mdl}.txt", "w") as f:
        for i in range(len(losses)):
            f.write(f"epoch: {i}, loss: {losses[i]}, acc: {accs[i]}\n")
    print("done")
