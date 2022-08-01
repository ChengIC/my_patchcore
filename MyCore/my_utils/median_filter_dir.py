import os
import cv2
from tqdm import tqdm

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']

def median_dir(input_dir, output_dir, filter_degree=5):
    print ('Start Apply Median Filter to {} with filter degree {}'.format(input_dir, filter_degree))
    for img_file in tqdm(os.listdir(input_dir)):
        if img_file.split('.')[-1] in IMG_FORMATS:
            img_file_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_file_path)
            median_img = cv2.medianBlur(img, filter_degree)
            output_img_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_img_path, median_img)
    print ('finish write median filer images from a folder')
