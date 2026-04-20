"""ResNet built on nanograd's Conv2d + BatchNorm2d + ReLU + residual connections.

``resnet18`` matches the ImageNet 224x224 layout.
``resnet_cifar`` is a smaller variant designed for 32x32 CIFAR images:
stem 3x3 conv + 3 stages, matching the "ResNet-20/32/44" family from the paper.
"""
from __future__ import annotations

from typing import List, Optional

from nanograd import nn
from nanograd.tensor import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        s = seed if seed is not None else 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, seed=s)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, seed=s + 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False, seed=s + 2),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = nn.F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return nn.F.relu(out + identity)


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        stem: str = "imagenet",
        base_channels: Optional[int] = None,
    ):
        super().__init__()
        # default widths: imagenet starts at 64 and doubles each stage; cifar at 16
        if base_channels is None:
            base_channels = 64 if stem == "imagenet" else 16
        self.in_channels = base_channels

        if stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(3, base_channels, 7, stride=2, padding=3, bias=False, seed=0),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        elif stem == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, base_channels, 3, stride=1, padding=1, bias=False, seed=0),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"unknown stem {stem}")

        widths = [base_channels * (2 ** i) for i in range(len(layers))]
        strides = [1] + [2] * (len(layers) - 1)

        seed_base = 100
        self.stages = []
        for i, (w, s, n) in enumerate(zip(widths, strides, layers)):
            stage = self._make_layer(block, w, n, stride=s, seed=seed_base)
            setattr(self, f"layer{i + 1}", stage)
            self.stages.append(stage)
            seed_base += 100

        final_channels = widths[-1] * block.expansion

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_channels, num_classes, seed=seed_base + 100)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int, seed: int) -> nn.Module:
        layers: list = [block(self.in_channels, out_channels, stride=stride, seed=seed)]
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, seed=seed + i * 10))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem="imagenet")


def resnet_cifar(num_blocks_per_stage: int = 3, num_classes: int = 10) -> ResNet:
    """ResNet-CIFAR style (e.g. ResNet-20 = n=3). 3 stages with 16/32/64 channels."""
    return ResNet(
        BasicBlock,
        [num_blocks_per_stage] * 3,
        num_classes=num_classes,
        stem="cifar",
        base_channels=16,
    )
