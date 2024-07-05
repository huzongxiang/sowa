from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class AnomalyCLIPDataModule(LightningDataModule):
    """LightningDataModule for Anormaly dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: Dict[str, str],
        dataset: Dict[str, Dataset],
        batch_size: int=16,
        image_size: int=336,
        num_workers: int=0,
        pin_memory: bool=False,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_data_dir = data_dir.train
        self.valid_data_dir = data_dir.valid
        self.test_data_dir = data_dir.test

        self.train_data_partial: Optional[Dataset] = dataset.train
        self.valid_data_partial: Optional[Dataset] = dataset.valid
        self.test_data_partial: Optional[Dataset] = dataset.test
        self.kshot_data_partial: Optional[Dataset] = dataset.kshot

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.kshot_data: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.train_data and not self.valid_data and not self.test_data:
            log.info("Instantiating train dataset!")
            self.train_data = self.train_data_partial(root=self.train_data_dir)
            log.info("Instantiating valid dataset!")
            self.valid_data = self.valid_data_partial(root=self.valid_data_dir)
            log.info("Instantiating test dataset!")
            self.test_data = self.test_data_partial(root=self.test_data_dir)
            if self.kshot_data_partial is not None:
                log.info("Instantiating kshot dataset!")
                self.kshot_data = self.kshot_data_partial(root=self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )
    
    def kshot_dataloader(self):
        if self.kshot_data is None:
            return None
        return DataLoader(
            dataset=self.kshot_data,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = AnomalyCLIPDataModule()
