import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Script to run a Keybored TorchScript model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument('script', type=Path, help="Path to the script to test.")
	parser.add_argument('images', type=Path, nargs='+', help="Image to process.")

	return parser.parse_args()

def main() -> None:
	args = parse_args()

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	script = torch.jit.load(args.script, map_location=device)

	transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize((400, 1008)),
		torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	for image_name in args.images:
		image = np.array(Image.open(image_name))
		input = transforms(image)
		input = input.to(device)

		mask = script(input.unsqueeze(0))

		mask = torch.nn.functional.interpolate(mask, image.shape[:2])[0, 0, ...]
		mask = mask.detach().cpu().numpy()
		mask = (mask > 0.5).astype(np.uint8)

		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
			rect = cv2.boundingRect(c)
			size = max(rect[2], rect[3]) // 2
			x = rect[0] + rect[2] // 2
			y = rect[1] + rect[3] // 2

			cv2.circle(image, (x, y), size, (252, 15, 192), -1, cv2.LINE_AA)
			cv2.circle(image, (x, y), size // 2, (55, 247, 19), -1, cv2.LINE_AA)

		# indices = np.where(mask != 0)
		# image = np.array(image)
		# image[indices[0], indices[1], :] = 0

		import matplotlib.pyplot as plt
		plt.imshow(image)
		plt.show()


if __name__ == '__main__':
	main()
