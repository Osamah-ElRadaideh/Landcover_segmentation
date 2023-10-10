from att_shufflenet import Unet
import torch 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import random
from sacred import Experiment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ex = Experiment('segmentation inference', save_git_info=False)

@ex.config
def defaults():
    output_name ='output.png'
    ckpt = "ckpt_best_loss.pth"



@ex.automain
def main(image_path, output_name, ckpt):
    model = Unet().to(device)
    states = torch.load(ckpt)
    model.load_state_dict(states)
    model.eval()

    img = cv2.imread(image_path).astype(np.float32)
    tensored = torch.from_numpy(img).permute(-1, 0, 1).unsqueeze(dim=0).to(device)
    with torch.inference_mode():
        segmented = model(tensored)
        masked = segmented.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        plt.imsave(output_name, masked)