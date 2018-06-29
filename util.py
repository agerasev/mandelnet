import numpy as np
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt


def convert_depth(depth_buffer, max_depth):
    return (255*np.log(np.maximum(depth_buffer, 0) + 1)/np.log(max_depth + 1)).astype(np.uint8)

def convert_color(color_buffer):
    return (255*np.clip(color_buffer, 0, 1)).astype(np.uint8)

def display_image(image):
    display(Image.fromarray((255*np.clip(image, 0, 1)).astype(np.uint8)))

def plot_images(data):
    if len(data.shape) == 3:
        plt.imshow(convert_color(data))
        plt.axis('off')
    elif len(data.shape) == 4:
        for i, img in enumerate(data):
            plt.subplot(1, len(data), i+1)
            plt.imshow(convert_color(img))
            plt.axis('off')
    elif len(data.shape) == 5:
        for j, seq in enumerate(data):
            for i, img in enumerate(seq):
                plt.subplot(len(seq), len(data), i*len(data)+j+1)
                plt.imshow(convert_color(img))
                plt.axis('off')
    plt.show()
