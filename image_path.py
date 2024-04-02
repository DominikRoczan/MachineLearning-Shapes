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