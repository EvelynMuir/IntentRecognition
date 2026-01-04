import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


class CutoutPIL:
    """Cutout data augmentation for PIL images.
    
    Randomly masks out a rectangular region from an image.
    """

    def __init__(self, cutout_factor: float = 0.5, fill_value: Optional[int] = None):
        """Initialize Cutout transform.

        :param cutout_factor: Factor to determine the size of the cutout region.
            The cutout size will be cutout_factor * image_size.
        :param fill_value: Value to fill the cutout region. If None, uses random RGB values.
        """
        self.cutout_factor = cutout_factor
        self.fill_value = fill_value

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply cutout to PIL image.

        :param img: PIL Image to apply cutout to.
        :return: PIL Image with cutout applied.
        """
        img_draw = ImageDraw.Draw(img)
        h, w = img.size[1], img.size[0]  # PIL uses (width, height)
        
        # Calculate cutout size
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        
        # Random center position
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)
        
        # Calculate bounding box
        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        
        # Fill color
        if self.fill_value is not None:
            fill_color = (self.fill_value, self.fill_value, self.fill_value)
        else:
            fill_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        
        # Draw rectangle
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        return img

