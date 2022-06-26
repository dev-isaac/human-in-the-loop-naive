from collections import OrderedDict
from io import BytesIO
from logging import Logger
from pathlib import Path
from logging import Logger
from typing import Any, Dict, List, Literal

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision.datasets.cifar import CIFAR10

import omegaconf
from tqdm import tqdm

from custom_datasets import (
    ModCIFAR10Small,
    ModCIFAR10Medium,
)

CLASSES_TO_IDX = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
IDX_TO_CLASSES = {val: key for key, val in CLASSES_TO_IDX.items()}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SubsetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        parent_dataset,
        additional_metadata: Dict[int, Any],
        indices,
        transforms,
    ) -> None:
        super().__init__()
        self.subset = torch.utils.data.Subset(parent_dataset, indices=indices)
        self.transforms = transforms
        self.metadata = additional_metadata

        self.reindexed_metadata = OrderedDict()
        for new_idx, orig_idx in enumerate(indices):
            self.reindexed_metadata[new_idx] = self.metadata[orig_idx]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int) -> Any:
        return (
            self.subset.__getitem__(index),
            CLASSES_TO_IDX[self.reindexed_metadata[index]],
        )


class Model:
    def __init__(
        self,
        logger: Logger = Logger(__name__),
        training_state: Literal[
            "untrained", "baseline", "fully-trained"
        ] = "untrained",
    ):
        self.logger = logger
        self.config = omegaconf.OmegaConf.load(
            Path(__file__).parent / "config.yaml"
        )
        self.training_state = training_state
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0")

        self.net: nn.Module = Net().to(device=self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.batch_size = 32

        ## Defining Torch Datasets
        self.dataset_path = Path(__file__).parent / "datasets"
        self.eval_set = CIFAR10(
            root=self.dataset_path,
            train=False,
            transform=self.transforms,
            download=True,
        )
        self.human_annotated_dataset = ModCIFAR10Medium
        self.train_set_medium = self.human_annotated_dataset(
            root=self.dataset_path, transform=self.transforms, download=True,
        )
        self.train_set_small = ModCIFAR10Small(
            root=self.dataset_path, transform=self.transforms, download=True
        )
        self.trainloader_baseline = torch.utils.data.DataLoader(
            self.train_set_small,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )
        self.trainloader_additional = torch.utils.data.DataLoader(
            self.train_set_medium,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

        ## Define fields tracking additional annotation
        self.accepted_id_min = 0
        self.accepted_id_max = len(self.train_set_medium)
        self.annotations: OrderedDict[int, str] = OrderedDict()

        ## Create copy of human-annotated dataset for serving images
        self.serving_dataset = self.human_annotated_dataset(
            self.dataset_path, download=False
        )
        self.next_annotation_idx: int = 0

    def commit_checkpoint(self, name: str) -> None:
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            Path(self.config.PATHS.CHECKPOINTS) / f"{name}.pt",
        )

    def list_checkpoints(self) -> List[str]:
        glob = Path(self.config.PATHS.CHECKPOINTS).glob("*.pt")
        return [p.name.replace(".pt", "") for p in glob]

    def load_checkpoint(self, name: str) -> None:
        ckpt = torch.load(Path(self.config.PATHS.CHECKPOINTS) / f"{name}.pt")

        self.net.load_state_dict(state_dict=ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def get_next_image(self):
        if self.next_annotation_idx >= self.accepted_id_max:
            return None

        pil_image, answer = self.serving_dataset[self.next_annotation_idx]

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        self.next_annotation_idx += 1

        return image_bytes, self.next_annotation_idx - 1, IDX_TO_CLASSES[answer]

    def train_baseline(self, epochs: int):
        self.net.train()
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(
                tqdm(
                    self.trainloader_additional,
                    desc=f"training... epoch {epoch} of {epochs}",
                ),
                0,
            ):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if self.use_gpu:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    self.logger.debug(
                        f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}"
                    )
                    running_loss = 0.0

            self.perform_eval()
        self.logger.info("Finished Training")

    def train_annotations(self) -> float:
        if self.annotations:
            self.net.train()

            indices = list(self.annotations.keys())

            subset_dataset = SubsetDataset(
                parent_dataset=self.train_set_medium,
                additional_metadata=self.annotations,
                indices=indices,
                transforms=self.transforms,
            )

            new_annotated_dataloader = torch.utils.data.DataLoader(
                subset_dataset, batch_size=self.batch_size,
            )

            running_loss = 0.0
            for i, data in enumerate(
                tqdm(
                    new_annotated_dataloader,
                    desc=f"training {len(subset_dataset)} new annotations...",
                ),
                0,
            ):
                # get the inputs; data is a list of [inputs, labels]
                (inputs, labels), new_annotation = data
                if self.use_gpu:
                    inputs = inputs.to(self.device)
                    new_annotation = new_annotation.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, new_annotation)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    self.logger.debug(
                        f"[{i + 1:5d}] loss: {running_loss / 2000:.3f}"
                    )
                    running_loss = 0.0

            if self.next_annotation_idx >= self.accepted_id_max:
                self.next_annotation_idx = 0
            self.annotations.clear()

        acc = self.perform_eval()
        return acc

    def perform_eval(self) -> float:
        self.net.eval()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(self.testloader, desc=f"evaluating..."):
                images, labels = data
                if self.use_gpu:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info(
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
        )
        print(self.net.state_dict())
        return 100 * correct / total

