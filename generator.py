import os
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate sample images for the Image Detection Challenge')
    return parser.parse_args()


def load_config():
    config_path = 'config_generator.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_random_rectangle(ny, nx, rect_min_height, rect_min_width, rect_max_scale):
    # Ensure there's room to generate a valid rectangle
    if ny <= rect_min_height or nx <= rect_min_width:
        raise ValueError('Image dimensions are too small for the specified minimum rectangle size.')

    # Choose the top-left corner such that there is at least rect_min_size remaining
    y0 = random.randint(0, ny - rect_min_height)
    x0 = random.randint(0, nx - rect_min_width)

    # Maximum possible height and width based on provided scale and image boundary
    max_height = min(ny - y0, int(ny * rect_max_scale))
    max_width  = min(nx - x0, int(nx * rect_max_scale))

    # In cases where the maximum available is less than the minimum, adjust it
    rect_height = rect_min_height if max_height < rect_min_height else random.randint(rect_min_height, max_height)
    rect_width  = rect_min_width if max_width < rect_min_width else random.randint(rect_min_width, max_width)

    y1 = y0 + rect_height
    x1 = x0 + rect_width
    return y0, x0, y1, x1


def add_noise(image_array, sigma):
    # Assume image_array is a numpy array with float values
    noise = np.random.normal(0, sigma, image_array.shape)
    noisy_image = image_array + noise
    # Clip to valid range [0, 255]
    np.clip(noisy_image, 0, 255, out=noisy_image)
    return noisy_image


def generate_image(config, image_index):
    # Choose an image size. config may provide a list of image_sizes (each as [ny, nx]).
    sizes = config.get('image_sizes')
    if not sizes:
        raise ValueError('Configuration must include image_sizes as a list of [ny, nx] pairs.')

    image_size = random.choice(sizes)
    ny, nx = image_size

    # Generate random colors for outer and inner rectangles
    outer_color = [random.randint(0, 255) for _ in range(3)]
    inner_color = [random.randint(0, 255) for _ in range(3)]

    # Create an image array and fill with the outer color
    image_arr = np.zeros((ny, nx, 3), dtype=np.float32)
    image_arr[:, :] = outer_color  

    # Get rectangle size constraints
    rect_min = config.get('rectangle_min_size', {'height': 10, 'width': 10})
    rect_min_height = rect_min.get('height', 10)
    rect_min_width  = rect_min.get('width', 10)

    rect_max_scale = config.get('rectangle_max_scale', 0.5)  # maximum fraction of the dimension

    # Generate a random rectangle within the image
    try:
        y0, x0, y1, x1 = generate_random_rectangle(ny, nx, rect_min_height, rect_min_width, rect_max_scale)
    except ValueError as e:
        print(f'Error generating rectangle: {e}')
        sys.exit(1)

    # Draw the rectangle (fill inner region) with inner_color
    image_arr[y0:y1, x0:x1] = inner_color

    # Optionally, apply noise if enabled
    noise_config = config.get('noise', {'enabled': False})
    if noise_config.get('enabled', False):
        sigma = noise_config.get('sigma', 10)
        image_arr = add_noise(image_arr, sigma)

    # After modifications, convert the array to uint8
    image_uint8 = image_arr.astype(np.uint8)
    
    # Return the image and the annotation data
    annotation = {
        'y0': y0,
        'x0': x0,
        'y1': y1,
        'x1': x1,
        'outer_color': outer_color,
        'inner_color': inner_color
    }
    return image_uint8, annotation


def main():
    args = parse_arguments()
    config = load_config()

    image_count = config.get('image_count', 10)
    output_folder = config.get('output_folder', 'images')
    ensure_dir(output_folder)

    annotations = []
    for i in range(image_count):
        img_array, ann = generate_image(config, i)
        filename = os.path.join(output_folder, f'image_{i:03d}.png')
        # Save using PIL
        im = Image.fromarray(img_array, mode='RGB')
        im.save(filename)
        ann['filename'] = filename
        annotations.append(ann)
        print(f'Generated {filename} with rectangle at (y0,x0)=({ann["y0"]},{ann["x0"]}) and (y1,x1)=({ann["y1"]},{ann["x1"]})')

    # Optionally, save the annotations to a json file
    ann_file = os.path.join(output_folder, 'annotations.json')
    with open(ann_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f'Annotations saved to {ann_file}')


if __name__ == '__main__':
    main()
