from typing import List, Any, Dict, Tuple, Union
import time
import numpy as np
from tabulate import tabulate
from scipy.ndimage import gaussian_filter

import torch
from torch import Tensor
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MetricCollection
from torchmetrics.classification import Accuracy, MulticlassAUROC

from src.models.components.functional import cosine_similarity_torch
from src.utils.metrics import image_level_metrics, pixel_level_metrics
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class AnomalyCLIPModule(LightningModule):
    """
    LightningModule for training an anomaly detection model using features extracted by a CLIP model.
    
    Attributes:
        net (nn.Module): The core model which contains the layers for feature extraction and processing.
        loss_focal (nn.Module): Focal loss function, helpful for handling class imbalance in datasets.
        loss_dice (nn.Module): Dice loss function, typically used for segmentation tasks.
        loss_ce (nn.Module): Cross-entropy loss for classification tasks.
    """

    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: Union[torch.nn.Module, Any],
        enable_validation: bool,
        compile: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        """
        Initialize a `AnomalyCLIPModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param loss: The loss function to use for training.
        :param enable_validation: Boolean to enable validation.
        :param compile: Boolean to enable compilation.
        :param kwargs: Additional keyword arguments.
        """
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

        # loss function
        self.loss_ce = loss["cross_entropy"]
        self.loss_focal = loss["focal"]
        self.loss_dice = loss["dice"]

        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=2),
                "auroc": MulticlassAUROC(num_classes=2),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.val_acc_best = MaxMetric()
        self.val_auroc_best = MaxMetric()

    def forward(self, images: Tensor, cls_name: str) -> Tuple:
        """
        Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tuple containing the model's outputs.
        """
        return self.net(images, cls_name)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # By default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks.
        self.val_metrics.reset()
        self.val_acc_best.reset()
        self.val_auroc_best.reset()

    def on_train_batch_start(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Hook called on the train batch start event."""
        # This function logs the learning rate at the start of each training bat
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True)  

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when training epoch end."""
        # by default lightning executes validation step sanity checks before training end.
        self.train_metrics.reset()

    def model_step(self, batch: Dict[str, Any]) -> Tuple:
        """
        Shared logic for training, validation, and testing using vectorized operations.

        :param batch: A dictionary containing the input data.
        :return: A tuple containing the model's predictions and targets.
        """
        images, masks, cls_name, labels = batch["image"], batch["image_mask"], batch["cls_name"], batch["anomaly"]
        _, _, _, _, anomaly_maps, text_probs = self.forward(images, cls_name)

        return text_probs, labels, anomaly_maps, masks

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dictionary) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A dictionary containing the computed losses.
        """
        logits, labels, anomaly_maps, masks = self.model_step(batch)
        loss_ce = self.loss_ce(logits, labels)

        masks = torch.where(masks > 0.0, 1, 0).squeeze(1)

        loss_focal = 0
        loss_dice = 0
        for i in range(len(anomaly_maps)):
            loss_focal += self.loss_focal(anomaly_maps[i], masks)
            loss_dice += self.loss_dice(anomaly_maps[i][:, 1, :, :], masks)
            loss_dice += self.loss_dice(anomaly_maps[i][:, 0, :, :], 1 - masks)

        loss = loss_ce + loss_focal + loss_dice

        self.log("train/loss_ce", loss_ce, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_focal", loss_focal, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_dice", loss_dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log_dict(self.train_metrics(logits, labels), on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss_ce": loss_ce,
            "loss_focal": loss_focal,
            "loss_dice": loss_dice,
            "loss": loss,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dictionary) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        if not self.hparams.enable_validation:
            return None

        logits, labels, anomaly_maps, masks = self.model_step(batch)
        loss_ce = self.loss_ce(logits, labels)

        masks = torch.where(masks > 0.0, 1, 0).squeeze(1)

        loss_focal = 0
        loss_dice = 0
        for i in range(len(anomaly_maps)):
            loss_focal += self.loss_focal(anomaly_maps[i], masks)
            loss_dice += self.loss_dice(anomaly_maps[i][:, 1, :, :], masks)
            loss_dice += self.loss_dice(anomaly_maps[i][:, 0, :, :], 1 - masks)

        loss = loss_ce + loss_focal + loss_dice

        self.log("val/loss_ce", loss_ce, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss_focal", loss_focal, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss_dice", loss_dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log_dict(self.val_metrics(logits, labels), on_step=False, on_epoch=True, prog_bar=True)

    def on_epoch_end(self) -> None:
        """Hook called at the end of an epoch."""
        # Custom actions at the end of each epoch
        pass

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Get current validation metrics
        if not self.hparams.enable_validation:
            return None
        
        current_metrics = self.val_metrics.compute()
        current_acc = current_metrics["val/acc"]
        current_auroc = current_metrics["val/auroc"]
        # update best so far val metrics
        self.val_acc_best.update(current_acc)
        self.val_auroc_best.update(current_auroc)
        # log `val_acc/auroc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/val_acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/val_roc_best", self.val_auroc_best.compute(), on_epoch=True, prog_bar=True)

        # reset
        self.val_metrics.reset()

    def kshot_step(self, kshot_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Compute features for the k-shot dataset.

        :param kshot_batch: A batch of data from the k-shot dataset.
        :param batch_idx: The index of the current batch.
        :return: A dictionary containing the computed features.
        """
        kshot_images = kshot_batch['image'].squeeze(0)        # Shape: [1, k_shot, C, H, W] -> [k_shot, C, H, W]
        cls_name = kshot_batch['cls_name'][0]                 # [str] -> str
        
        k_shot = kshot_images.shape[0]
        classnames = kshot_batch['cls_name'] * k_shot         # [str] -> [k_shot * str]

        _, _, kshot_patches, _, _, _ = self.forward(kshot_images, classnames)    # List[(k_shot, L, C)]

        return {cls_name: kshot_patches}

    def on_test_start(self) -> None:
        """Lightning hook that is called when test begins."""
        self.results = {}
        self.mem_features = {}
        device = self.device if hasattr(self, 'device') else torch.device('cpu')
        if self.hparams.k_shot:
            for batch_idx, kshot_batch in enumerate(self.trainer.datamodule.kshot_dataloader()):
                kshot_batch =  {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in kshot_batch.items()}
                self.mem_features.update(self.kshot_step(kshot_batch, batch_idx))

    def kshot_anomaly(self, patch_tokens: List[Tensor], cls_name: str) -> Tensor:
        """
        Compute the k-shot anomaly maps for a given set of patch tokens.

        :param patch_tokens: List of tensors, where each tensor represents the patch tokens for a single image.
                            Each tensor has shape (L, C) where L is the number of patches and C is the feature dimension.
        :param cls_name: String representing the class name used to retrieve the corresponding k-shot patches.
        :return: A tensor representing the aggregated k-shot anomaly map for the given set of patch tokens.
                The returned tensor has shape (1, H, W) where H and W are the dimensions of the resized anomaly map.
        """
        kshot_patch = self.mem_features[cls_name]
        B, L, C = kshot_patch[0].shape
        H = int(L ** 0.5)
        anomaly_maps_kshot = []

        for idx, patch in enumerate(patch_tokens):
            kshot_patch_expand = kshot_patch[idx].reshape(B * L, C)
            cosine_similarity = cosine_similarity_torch(kshot_patch_expand, patch)
            cosine_distance = (1. - cosine_similarity).min(dim=0)[0]
            anomaly_map_kshot = cosine_distance.reshape(1, 1, H, H)   
            anomaly_map_kshot = F.interpolate(anomaly_map_kshot,
                                        size=self.net.image_size, mode='bilinear', align_corners=True)
            anomaly_maps_kshot.append(anomaly_map_kshot[0])
        
        anomaly_maps_kshot = torch.stack(anomaly_maps_kshot).sum(dim=0)
        return anomaly_maps_kshot

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dictionary) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        :return: A dictionary containing the anomaly maps.
        """
        images, masks, classnames, labels = batch["image"], batch["image_mask"], batch["cls_name"], batch["anomaly"]
        _, _, patches, _, anomaly_maps, text_probs = self.forward(images, classnames)

        self.log_dict(self.test_metrics(text_probs, labels), on_step=False, on_epoch=True, prog_bar=True)

        masks = torch.where(masks > 0.5, 1, 0).squeeze(1)
        batch_size = images.size(0)

        anomaly_maps = torch.stack(anomaly_maps)[:, :, 1, :, :].sum(dim=0)
        if self.hparams.filter:
            anomaly_maps = torch.stack([torch.from_numpy(gaussian_filter(anomaly_map, sigma=4)) for anomaly_map in anomaly_maps.detach().cpu()], dim=0)

        updated_anomaly_maps = []
        for i in range(batch_size):
            cls_key = classnames[i]
            if cls_key not in self.results:
                self.results[cls_key] = {
                    'imgs_masks': [],
                    'anomaly_maps': [],
                    'gt_sp': [],
                    'pr_sp': []
                }

            mask = masks[i]
            anomaly_map = anomaly_maps[i]

            if self.hparams.k_shot:
                patch_tokens = [patch[i] for patch in patches]
                anomaly_maps_kshot = self.kshot_anomaly(patch_tokens, cls_key)
                anomaly_map = anomaly_map.cpu() + anomaly_maps_kshot.cpu()

            self.results[cls_key]['imgs_masks'].append(mask)  # Store masks for pixel-level metrics
            self.results[cls_key]['anomaly_maps'].append(anomaly_map)
            
            self.results[cls_key]['gt_sp'].append(labels[i].cpu().numpy().item())  # Store labels for image-level metrics
            self.results[cls_key]['pr_sp'].append(text_probs[i][1].detach().cpu().numpy().item())  # Store predicted probabilities

            updated_anomaly_maps.append(anomaly_map)

        updated_anomaly_maps = torch.stack(updated_anomaly_maps, dim=0).squeeze(1)

        return {
            "anomaly_maps": updated_anomaly_maps,
            "abnormal": text_probs,
        }

    def on_test_epoch_end(self) -> None:
        """
        Lightning hook that is called when a test epoch ends.

        This method processes the results collected during the test epoch, computes various metrics
        (image-level AUROC, image-level AP, pixel-level AUROC, pixel-level AUPRO), and logs these metrics.
        It also formats the metrics into a table and logs the table. Finally, it resets the results
        to ensure a clean state for the next epoch.

        Operations performed in this method:
        1. Start a timer to measure the duration of result processing.
        2. Initialize lists and a table for storing metrics for each class.
        3. Loop over the results for each class:
        a. Convert the stored masks and anomaly maps to numpy arrays.
        b. Compute the image-level AUROC, image-level AP, pixel-level AUROC, and pixel-level AUPRO for each class.
        c. Append the computed metrics to the corresponding lists and the table.
        4. Compute the mean values for each metric across all classes.
        5. Log the mean metrics.
        6. Format the metrics into a table using the `tabulate` library and log the table.
        7. Log the duration of result processing.
        8. Reset the results to ensure a clean state for the next epoch.
        """

        start_time = time.time()
        log.info(f"Processing test results...")

        tables = []
        image_auroc_list = []
        image_ap_list = []
        image_f1_list = []
        pixel_auroc_list = []
        pixel_ap_list = []
        pixel_f1_list = []
        pixel_aupro_list = []

        for cls_key, data in self.results.items():
            table = [cls_key]
            data['imgs_masks'] = torch.stack(data['imgs_masks']).detach().cpu().numpy()
            data['anomaly_maps'] = torch.stack(data['anomaly_maps']).detach().cpu().numpy()

            # Handling few-shot mode
            if self.hparams.k_shot:
                pr_sp_tmp = np.array([np.max(anomaly_map) for anomaly_map in data["anomaly_maps"]])
                pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
                pr_sp = 0.5 * (np.array(data["pr_sp"]) + pr_sp_tmp)

                data["pr_sp"] = pr_sp

            image_auroc = image_level_metrics(self.results, cls_key, "image-auroc")
            image_ap = image_level_metrics(self.results, cls_key, "image-ap")
            image_f1 = image_level_metrics(self.results, cls_key, "image-f1-max")
            pixel_auroc = pixel_level_metrics(self.results, cls_key, "pixel-auroc")
            pixel_ap = pixel_level_metrics(self.results, cls_key, "pixel-ap")
            pixel_f1 = pixel_level_metrics(self.results, cls_key, "pixel-f1-max")
            pixel_aupro = pixel_level_metrics(self.results, cls_key, "pixel-aupro")

            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            table.append(str(np.round(image_f1 * 100, decimals=1)))
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_ap * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            tables.append(table)

            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            image_f1_list.append(image_f1)
            pixel_auroc_list.append(pixel_auroc)
            pixel_ap_list.append(pixel_ap)
            pixel_f1_list.append(pixel_f1)
            pixel_aupro_list.append(pixel_aupro)

        mean_image_auroc = np.mean(image_auroc_list)
        mean_image_ap = np.mean(image_ap_list)
        mean_image_f1 = np.mean(image_f1_list)
        mean_pixel_auroc = np.mean(pixel_auroc_list)
        mean_pixel_ap = np.mean(pixel_ap_list)
        mean_pixel_f1 = np.mean(pixel_f1_list)
        mean_pixel_aupro = np.mean(pixel_aupro_list)
        objective = (mean_image_auroc + mean_image_f1 + mean_image_ap + mean_pixel_auroc + mean_pixel_ap + mean_pixel_f1 + mean_pixel_aupro) / 7

        tables.append(
                [
                    'mean', 
                    str(np.round(mean_image_auroc * 100, decimals=1)),
                    str(np.round(mean_image_ap * 100, decimals=1)), 
                    str(np.round(mean_image_f1 * 100, decimals=1)), 
                    str(np.round(mean_pixel_auroc * 100, decimals=1)),
                    str(np.round(mean_pixel_ap * 100, decimals=1)),
                    str(np.round(mean_pixel_f1 * 100, decimals=1)),
                    str(np.round(mean_pixel_aupro * 100, decimals=1)),
                ]
        )

        # Log the mean metrics
        self.log("test/image_auroc", mean_image_auroc, on_epoch=True, prog_bar=True)
        self.log("test/image_ap", mean_image_ap, on_epoch=True, prog_bar=True)
        self.log("test/image_f1", mean_image_f1, on_epoch=True, prog_bar=True)
        self.log("test/pixel_auroc", mean_pixel_auroc, on_epoch=True, prog_bar=True)
        self.log("test/pixel_ap", mean_pixel_ap, on_epoch=True, prog_bar=True)
        self.log("test/pixel_f1", mean_pixel_f1, on_epoch=True, prog_bar=True)
        self.log("test/pixel_aupro", mean_pixel_aupro, on_epoch=True, prog_bar=True)
        self.log("test/objective", objective, on_epoch=True, prog_bar=True)

        metrics = tabulate(tables, headers=["objects", "image_auroc", "image_ap", "image_f1", "pixel_auroc", "pixel_ap", "pixel_f1", 'pixel_aupro'], tablefmt="pipe")
        
        end_time = time.time()
        duration = end_time - start_time
        log.info(f"Processed test results in {duration:.2f} seconds.")
        log.info(f"\n{metrics}")

        # Reset results to clean state after test
        self.results = {}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
