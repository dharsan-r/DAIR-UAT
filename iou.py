import torch
from PIL import Image
import csv


def image_to_matrix(image_path):
    # print("insde image to matrix")
    image = Image.open(image_path)
    image = image.convert('RGB')
    width, height = image.size
    pixels = torch.zeros((height , width,5), dtype=torch.float32)

    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            pixels[y, x] = torch.tensor([x, y, r, g, b])
    
    # print("Pixels: ", pixels)
    return pixels

def filter_rgb_range(matrix, rgb_value):
  
    # print("matrix", matrix)
    rgb_range = {
        'R': (rgb_value['R'] - 10, rgb_value['R'] + 10),
        'G': (rgb_value['G'] - 10, rgb_value['G'] + 10),
        'B': (rgb_value['B'] - 10, rgb_value['B'] + 10)
        }
    mask = ((matrix[:, :, 2] >= rgb_range['R'][0]) & (matrix[:, :, 2] <= rgb_range['R'][1]) &
            (matrix[:, :, 3] >= rgb_range['G'][0]) & (matrix[:, :, 3] <= rgb_range['G'][1]) &
            (matrix[:, :, 4] >= rgb_range['B'][0]) & (matrix[:, :, 4] <= rgb_range['B'][1]))
    # print("mask",mask)
    filtered_matrix = matrix.clone()
    filtered_matrix[~mask] = 0
    # print("filtered", filtered_matrix)
    return filtered_matrix



# Example usage
def read_rgb_table_from_csv(csv_file_path):
    """
    Read the RGB table from a CSV file and convert it into a dictionary.

    Parameters:
    csv_file_path (str): Path to the CSV file

    Returns:
    dict: Dictionary where keys are object names and values are dictionaries with keys 'R', 'G', 'B'
    """
    rgb_table = {}
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            object_name = row['name']
            rgb_table[object_name] = {
                'R': (int(row['r'])),
                'G': (int(row['g'])),
                'B': (int(row['b']))
            }
    return rgb_table



def calculate_iou_rgb(set1, set2, object_rgb_range):
    """
    Calculate the Intersection over Union (IoU) of two sets of (x, y, RGB) values.

    Parameters:
    set1 (tensor): Matrix of shape (height, width, 5) where each element contains [x, y, R, G, B]
    set2 (tensor): Matrix of shape (height, width, 5) where each element contains [x, y, R, G, B]
    object_rgb_range (dict): Dictionary with keys 'R', 'G', 'B' each containing a tuple (min, max) representing the range of RGB values

    Returns:
    float: IoU value
    """
    # Filter the sets to only include the object RGB values within the specified range
    
    set1 = filter_rgb_range(set1, object_rgb_range)
    set2 = filter_rgb_range(set2, object_rgb_range)

    # Compute the intersection
    intersection = torch.logical_and(set1[:, :, 2:] != 0, set2[:, :, 2:] != 0)
    intersection_area = torch.sum(intersection).item()
    print("Intersection Area: ", intersection_area)

    # Compute the area of each set
    set_union = torch.logical_or(set1[:, :, 2:] != 0, set2[:, :, 2:] != 0)
    set_union_area = torch.sum(set_union).item()
    print("Union Area: ", set_union_area)

    # Compute the IoU
    if set_union_area == 0:
        return float('nan')  # Avoid division by zero
    iou = intersection_area / set_union_area

    return iou  
    

object_name = 'Car'
rgb_table = read_rgb_table_from_csv('/home/saleh/QMIND/class_dict.csv')
rgb_values = rgb_table.get(object_name, None)

# debugging purposes
# rgb_values = {'R': 35, 'G': 223, 'B': 11}

print(f"RGB values for {object_name}: {rgb_values}")
set1 = image_to_matrix('/home/saleh/QMIND/Images and masks/masked_image.png')
set2 = image_to_matrix('/home/saleh/QMIND/Images and masks/masked_image_altered.png')
iou = calculate_iou_rgb(set1, set2, rgb_values)
print(f"IoU: {iou}")