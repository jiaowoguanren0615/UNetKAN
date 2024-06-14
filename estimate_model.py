import os
import time
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import UKAN_samll, UKAN_base, UKAN_large


@torch.no_grad()
def predictor(model, img, mask, device):
    model.eval()

    img = img.to(device)
    output = model(img)

    prediction = output.argmax(1).squeeze(0)
    prediction = prediction.to("cpu").numpy().astype(np.uint8)

    prediction[prediction == 1] = 255

    prediction[mask == 0] = 0
    return prediction


def run_pred(args, model, weights_path, img_path, roi_mask_path):
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = args.device
    print("using {} device.".format(device))

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    prediction = predictor(model, img, roi_img, device)

    mask = Image.fromarray(prediction)
    mask.save("./test_result.png")


# if __name__ == '__main__':
#     run_pred()