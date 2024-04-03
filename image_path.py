<<<<<<< HEAD
image_path = 'new_Image/107.jpg'


def print_allpixels():
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    print(image)
    print(image.shape)

    with open('all_pxels.txt', 'w') as file:
        for row in image:
            np.savetxt(file, row)

if __name__ == '__main__':
    print_allpixels()
=======
image_path = 'new_Image/107.jpg'

import cv2 as cv2
import numpy as np


def print_allpixels(image_path):
    """
    Reads an image from the specified path, prints its pixel values and shape, and saves all pixel values to a text file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """

    image = cv2.imread(image_path)
    print(image)
    print(image.shape)

    with open('all_pxels.txt', 'w') as file:
        for row in image:
            np.savetxt(file, row)


if __name__ == '__main__':
    print_allpixels(image_path)
>>>>>>> acf6128c69a0297734e9fe79087dc8bcf096cec3
