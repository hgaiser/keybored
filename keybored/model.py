from typing import List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class KeyboredSegmentation(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = smp.DeepLabV3Plus()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.sigmoid(self.model(x))

	def training_step(self, batch: List[torch.Tensor], _batch_index: int) -> torch.Tensor:
		input, target = batch

		prediction = self.model(input)
		import numpy as np
		np.save('prediction.npy', torch.sigmoid(prediction).detach().cpu().numpy())

		loss = smp.losses.DiceLoss('multilabel')(prediction, target) + smp.losses.FocalLoss('multilabel')(prediction, target)
		self.log('loss', loss)

		return loss

	def configure_optimizers(self) -> torch.optim.Optimizer:
		return torch.optim.SGD(
			self.parameters(),
			lr=0.2,
			momentum=0.9,
			weight_decay=1e-4,
		)

