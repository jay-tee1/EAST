from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import math
import requests
import json
import numpy as np
import os

#define image the OCR should be performed on with setting IMAGE_PATH accordingly
IMAGE_PATH = './DJI_0244_7257.jpeg'
TXT_FILE = os.path.splitext(IMAGE_PATH)[0] + '.txt'

#find rotation angle to axis-align the bounding box, using the slope
def find_rotation_angle(coordinates):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)

    slope = (mid2[1] - mid1[1]) / (mid2[0] - mid1[0])

    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


#crop image to min and max values of x and y
def crop(img, box_coordinates):
    x1, y1, x2, y2, x3, y3, x4, y4 = box_coordinates

    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)

    cropped_img = img.crop((min_x, min_y, max_x, max_y))

    return cropped_img



#create a mask for the bounding box and set everything else to white
def draw_mask(image_path, box_coordinates):
    img = Image.open(image_path)
    
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(box_coordinates, outline=1, fill=255)
    
    #puzzle together resulting image with mask
    result = Image.new("RGB", img.size, (255, 255, 255))
    result.paste(img, mask=mask)

    return result

#find the first pixel values != white as in mask
def get_colored_bbox(image):
    img_array = np.array(image)

    non_white_pixels = np.any(img_array != [255, 255, 255], axis=-1)

    #get min and max x and y values to later crop image
    rows, cols = np.where(non_white_pixels)
    bbox = (min(cols), min(rows), max(cols), max(rows))

    return bbox
    


#process per bounding box (ID)
def process_box(box, i):
    image = draw_mask(IMAGE_PATH, box)
    image = crop(image, box)
    rotation_angle = find_rotation_angle(box)
    #print("angle  ", rotation_angle)
    rotated_image = image.rotate(rotation_angle+90, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    bbox = get_colored_bbox(rotated_image)
    cropped_image = rotated_image.crop(bbox)
    cropped_image.show()
    #cropped_image.save(f'image_{i}.jpg')
    predict_ids(cropped_image)

#get all predicted bounding boxes out of one text file
def get_boxes(path):
    boxes = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                i+=1
                stripped_line = line.strip()
                if stripped_line and not stripped_line.isspace(): 
                    values = stripped_line.split(',')[:8]
                    box = list(map(int, values))
                    #start pre-processing and OCR for every box
                    process_box(box, i)
                    box = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
                    boxes.append(box)
    except FileNotFoundError:
        return []
    return boxes

#trOCR usage
def predict_ids(image):

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


    #print("=====")
    #print(generated_ids)
    print("recognized text:  ", f"'{generated_text}'")

boxes = get_boxes(TXT_FILE)
