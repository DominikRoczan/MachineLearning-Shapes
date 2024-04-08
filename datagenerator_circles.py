from PIL import Image, ImageDraw
import random
import os
import shutil


def generate_background_with_circles(width, height, num_jpg_circles, destination_path_circles):
    """
        Generates images with random circles and discs.

        Args:
            width (int): Width of the images.
            height (int): Height of the images.
            num_jpg_circles (int): Number of images to generate.
            destination_path_circles (str): Path to save the generated images.
        """

    for i in range(num_jpg_circles):
        # Random background color for each image
        background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new('RGB', (width, height), color=background_color)
        draw = ImageDraw.Draw(image)

        # Number of circles and discs in the image
        num_circles = 1

        # Drawing circles and discs
        for _ in range(num_circles):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            radius = random.randint(10, min(width, height) // 2)
            x2 = x1 + radius
            y2 = y1 + radius

            # Random fill and outline for the circle or disc
            fill_color = random.choice([None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])
            draw.ellipse([x1, y1, x2, y2], fill=fill_color, outline=fill_color)

        # Saving the image
        image.save(os.path.join(destination_path_circles, f'CIRCLES_{i + 1}.jpg'), format='JPEG')


def generate_multiple_images(width, height, num_jpg_circle, source_path):
    """
    Generates multiple images.

    Args:
        width (int): Width of the images.
        height (int): Height of the images.
        num_jpg_circle (int): Number of images to generate.
        source_path (str): Path to save the generated images.
    """
    os.makedirs(source_path, exist_ok=True)
    generate_background_with_circles(width, height, num_jpg_circle, source_path)


def storage(num_jpg_circle, source_path, destination_path_circles):
    """
    Moves generated images to a destination folder.

    Args:
        num_jpg_circle (int): Number of images.
        source_path (str): Path of the source folder.
        destination_path_circles (str): Path of the destination folder.
    """
    os.makedirs(destination_path_circles, exist_ok=True)
    for i in range(num_jpg_circle):
        source_file = f'CIRCLES_{i + 1}.jpg'
        try:
            shutil.copy(os.path.join(source_path, source_file), os.path.join(destination_path_circles, source_file))
        except shutil.SameFileError:
            # Ignore SameFileError
            pass


if __name__ == "__main__":
    width = 224
    height = 224
    num_jpg_circle = 6
    source_path = 'Generator'
    destination_path_circles = os.path.join(source_path, 'Circles')

    generate_multiple_images(width, height, num_jpg_circle, destination_path_circles)
    storage(num_jpg_circle, destination_path_circles, destination_path_circles)
