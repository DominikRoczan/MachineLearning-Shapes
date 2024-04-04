from PIL import Image, ImageDraw
import random
import os
import shutil


def generate_background_with_triangles(width, height, num_jpg_tr, destination_path_triangles):
    """
    Generates images with random triangles.

    Args:
        width (int): Width of the images.
        height (int): Height of the images.
        num_jpg_tr (int): Number of images to generate.
        destination_path_triangles (str): Path to save the generated images.
    """
    for i in range(num_jpg_tr):
        # Random background color for each image
        background = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new('RGB', (width, height), color=background)
        draw = ImageDraw.Draw(image)

        # Number of triangles in the image
        num_triangles = 1

        # Drawing triangles
        for _ in range(num_triangles):
            points = []
            # Random coordinates of triangle vertices
            for _ in range(3):
                x = random.randint(0, width)
                y = random.randint(0, height)
                points.append((x, y))

            # Random fill and outline for the triangles
            fill_color = random.choice([None, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])
            draw.polygon(points, fill=fill_color, outline=fill_color)

        # Saving the image
        image.save(os.path.join(destination_path_triangles, f'TRIANGLES_{i + 1}.jpg'), format='JPEG')


def generate_multiple_images(width, height, num_jpg_tr, destination_path):
    """
    Generates multiple images.

    Args:
        width (int): Width of the images.
        height (int): Height of the images.
        num_jpg_tr (int): Number of images to generate.
        destination_path (str): Path to save the generated images.
    """
    os.makedirs(destination_path, exist_ok=True)
    generate_background_with_triangles(width, height, num_jpg_tr, destination_path)


def storage(ilosc_jpg_tr, source_path, destination_path_triangles):
    # Tworzenie katalogu docelowego
    os.makedirs(destination_path_triangles, exist_ok=True)
    for i in range(ilosc_jpg_tr):
        source_file = f'TRIANGLES_{i + 1}.jpg'
        try:
            shutil.copy(os.path.join(source_path, source_file), os.path.join(destination_path_triangles, source_file))
        except shutil.SameFileError:
            # Ignorowanie błędu SameFileError
            pass


width = 224
height = 224
num_jpg_tr = 40
source_path = 'Generator'
destination_path_triangles = os.path.join(source_path, 'Triangles')

if __name__ == "__main__":
    generate_multiple_images(width, height, num_jpg_tr, destination_path_triangles)
    storage(num_jpg_tr, destination_path_triangles, destination_path_triangles)
    storage(num_jpg_tr, destination_path_triangles, destination_path_triangles)

