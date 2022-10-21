import argparse
from typing import Tuple
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

from keybored import model as km


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Training script for the Keybored segmentation network.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('input_dir', type=Path, help="Path to directory containing input images. This directory is expected to have an 'images' and 'masks' subdir.")

	return parser.parse_args()


class SaveScript(pl.Callback):
	def on_train_epoch_end(self, _trainer: pl.Trainer, model: km.KeyboredSegmentation):
		script = model.to_torchscript()
		script.save('script.pt')


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, input_dir: Path):
		self.input_dir = input_dir
		self.images = []
		self.masks = []
		self.transforms = A.Compose([
			# A.RandomRotate90(),
			# A.HorizontalFlip(p=0.5),
			# A.VerticalFlip(p=0.5),
			A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
			A.Blur(blur_limit=3),
			A.OpticalDistortion(),
			A.GridDistortion(),
			A.HueSaturationValue(),
			A.RandomBrightnessContrast(p=0.2),
			A.Resize(400, 1008),
			A.Normalize(),
			ToTensorV2(),
		])

		for file in (input_dir / 'images').iterdir():
			if file.suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
				image = Image.open(file).convert('RGB')
				mask = Image.open(input_dir / 'masks' / file.name).convert('L')

				self.images.append(image)
				self.masks.append(mask)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		image = self.images[index]
		mask = self.masks[index]

		transformed = self.transforms(image=np.array(image), mask=(np.array(mask) > 0).astype(np.uint8))

		return transformed['image'], transformed['mask'].unsqueeze(0)

	def __len__(self) -> int:
		return len(self.images)


def main() -> None:
	args = parse_args()

	model = km.KeyboredSegmentation()

	train_dataset = ImageDataset(args.input_dir)
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=2,
		drop_last=True,
		num_workers=8,
		shuffle=True,
	)

	trainer = pl.Trainer(
		accelerator='auto',
		callbacks=[
			# pl.callbacks.ModelCheckpoint(dirpath='models', save_top_k=1, monitor='loss', save_weights_only=True),
			SaveScript(),
		],
		log_every_n_steps=4,
		max_epochs=500,
	)
	trainer.fit(
		model=model,
		train_dataloaders=train_dataloader,
	)

if __name__ == '__main__':
	main()
