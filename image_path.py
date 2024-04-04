image_path = 'Generator/Circles/CIRCLES_6.jpg'

import cv2
import numpy as np

def print_allpixels():


    image = cv2.imread(image_path)
    print(image)
    print(image.shape)

    with open('all_pxels.txt', 'w') as file:
        for row in image:
            np.savetxt(file, row)

if __name__ == '__main__':
    print_allpixels()