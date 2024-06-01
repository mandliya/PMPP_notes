# code to generate colored pixel art with random colors
from PIL import Image, ImageDraw, ImageFilter
import random


def generate_colored_image(width: int, height: int, path: str):
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    for i in range(0, width, 10):
        for j in range(0, height, 10):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([i, j, i + 10, j + 10], fill=color)
    image.save(path)

def convert_rgb_to_gray(image_path: str, output_path: str):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    gray_image.save(output_path)

def combine_side_by_side(image1_path: str, image2_path: str, output_path: str):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    width, height = image1.size
    new_image = Image.new('RGB', (2 * width + 20, height))
    new_image.paste(image1, (0, 0))
    # add some whitespace between the images
    new_image.paste(image2, (width + 20, 0))
    new_image.save(output_path)

def blur_image(image_path: str, output_path: str):
    image = Image.open(image_path)
    blurred_image = image.filter(ImageFilter.BLUR)
    blurred_image.save(output_path)


if __name__ == '__main__':
    colored_image_path = 'images/generated_colored_image.png'
    gray_image_path = 'images/generated_gray_image.png'
    combined_image_path = 'images/rgb_to_grayscale.png'
    blur_image_path = 'images/blur_image.png'
    color_blur_image_path = 'images/color_blur_image.png'
    generate_colored_image(200, 200, colored_image_path)
    print('Colored image generated successfully at path:', colored_image_path)
    convert_rgb_to_gray(colored_image_path, gray_image_path)
    print('Image converted to grayscale successfully at path:', gray_image_path)
    combine_side_by_side(colored_image_path, gray_image_path, combined_image_path)
    print('Images combined successfully at path:', combined_image_path)
    blur_image(colored_image_path, color_blur_image_path)
    print('Image blurred successfully at path:', blur_image_path)
    combine_side_by_side(colored_image_path, color_blur_image_path, blur_image_path)
    