

import cv2
import numpy as np

obj_img_path = 'datasets/full_body/test/objs/D_P_F1_SS_V_B_MD_V_B_back_0907140704.jpg'
obj_img = cv2.imread(obj_img_path)
w,h,c= obj_img.shape
resize_img = cv2.resize(obj_img,[int(h/2),int(w/2)])
median = cv2.medianBlur(resize_img, 3)


concat_img = np.concatenate((resize_img, median), axis=1)



normal_img_path = 'datasets/full_body/train/good/D_N_F1_CL_V_LA_LW_V_RA_back_0907141138.jpg'
normal_img = cv2.imread(normal_img_path)
resize_img2 = cv2.resize(normal_img,[int(h/2),int(w/2)])
median2 = cv2.medianBlur(resize_img2, 3)

concat_img2 = np.concatenate((resize_img2, median2), axis=1)

concat_img3 = np.concatenate((concat_img, concat_img2), axis=0)

window_name = 'image'
cv2.imshow(window_name, concat_img3)

cv2.waitKey(0)