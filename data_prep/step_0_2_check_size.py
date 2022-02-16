
from __future__ import annotations
import os
import cv2


cropped_folder = './datasets/THz_Body/cropped_objs'
output_folder = './datasets/THz_Body/test/objs'
if not os.path.exists(output_folder):
    os.makedirs (output_folder)

av_s0 = 735
av_s1 = 280
for img_filename in os.listdir(cropped_folder):
    img_path = os.path.join(cropped_folder,img_filename)
    if '.jpg' in img_path:
        img = cv2.imread(img_path)
        res = cv2.resize(img,(280, 735), interpolation = cv2.INTER_CUBIC)
        output_img_path = os.path.join(output_folder,img_filename)
        cv2.imwrite(output_img_path, res)

        

