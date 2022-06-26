from typing import Callable, Optional
from torchvision.datasets.cifar import CIFAR10


class ModCIFAR10Medium(CIFAR10):
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root=root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class ModCIFAR10Small(CIFAR10):
    train_list = [
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(
            root=root,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
