import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import os
import torch
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfl_techniques import sfl

class SaliencyMapVisualizer:
    def __init__(self, img_path, device='cuda'):
        # Initialize and compute the relevance scores
        self.original_image = Image.open(img_path).resize((224, 224)).convert('L')
        self.original_image_array = np.array(original_image)

        # Retrieve the saved scores and values from the RelevanceScore instance
        self.Ep = relevance_score_instance.Ep
        self.Ef = relevance_score_instance.Ef
        self.Np = relevance_score_instance.Np
        self.Nf = relevance_score_instance.Nf


    def visualize_pixel_scores(self, dataset):
        H, W = (224,224)
        scores = {
            'Ep': np.zeros((H, W)),
            'Ef': np.zeros((H, W)),
            'Np': np.zeros((H, W)),
            'Nf': np.zeros((H, W)),
            'ochiai': np.zeros((H, W)),
            'tarantula': np.zeros((H, W)),
            'zoltar': np.zeros((H, W)),
            'wong1': np.zeros((H, W))
        }
        
        for pixel in dataset:
            i, j = pixel['position']
            for score_type in scores.keys():
                scores[score_type][i, j] = pixel[score_type]
        
        # Invert values for ochiai and tarantula
        scores['ochiai'] = 1 - scores['ochiai']
        scores['tarantula'] = 1 - scores['tarantula']
        scores['zoltar']  = 1 - scores['zoltar']
        scores['wong1']  = 1 - scores['wong1']
        
        # Set up the plot
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("Pixel-wise Scores Visualization", fontsize=16)
        # Define a color map
        cmap = plt.get_cmap('jet')
        # Plot each score type
        for idx, (score_type, score_data) in enumerate(scores.items()):
            row = idx // 4
            col = idx % 4
            ax = axs[row, col]
            
            # Create the heatmap
            im = ax.imshow(self.original_image_array, cmap='gray', alpha=1)
            im = ax.imshow(score_data, cmap=cmap, alpha=0.5)
            ax.set_title(score_type)
            ax.axis('off')
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        # Adjust layout and display
        plt.tight_layout()
        plt.show()