from my_utils.inference_utils import *
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.pyplot import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import xml.etree.ElementTree as ET
from PIL import Image
import statistics

def get_median_val(inputlist):    
    return statistics.median(inputlist)

def get_sum_counts(pixels_information):
    sum_list = []
    for fv_info in pixels_information:
        sum_count = 0
        for k in fv_info:
            sum_count += fv_info[k]
        sum_list.append(sum_count)
    return sum_list

# def filter_half_median(rank_list):
#     median_val = get_median_val(rank_list)
#     filtered_list = []
#     for r in rank_list:
#         if r >= median_val:
#             filtered_list.append(r)
#         else:
#             filtered_list.append(0)
#     return filtered_list

def filter_quantile(rank_list,th=0.75):
    th_val = np.quantile(rank_list, th)
    filtered_list = []
    for r in rank_list:
        if r >= th_val:
            filtered_list.append(r)
        else:
            filtered_list.append(0)
    return filtered_list


def assign_weights_by_small(rank_list):
    rev_rank_list = [min(rank_list)/float(i) for i in rank_list]
    rev_rank_list = filter_quantile(rev_rank_list)
    norm = [(float(i)-min(rev_rank_list))/(max(rev_rank_list)-min(rev_rank_list)) for i in rev_rank_list]
    adjust = [float(i)/sum(norm) for i in norm]
    return adjust

def avg_by_small (pixels_information, all_fv_imgs):
    rank_list = get_sum_counts(pixels_information)
    weight_list = assign_weights_by_small(rank_list)
    print (weight_list)
    fv = weight_list[0] * all_fv_imgs[0]
    for idx, w in enumerate(weight_list):
         if idx>=1:
            fv += w*all_fv_imgs[idx]
    return fv

def getBBox(annotation_dir, img_id):
    xml_filePath = os.path.join (annotation_dir, img_id + '.xml')
    xml_file = open(xml_filePath, encoding='UTF-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls != 'HUMAN':
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)
            single_box = {
                    'xmin':xmin,
                    'xmax':xmax,
                    'ymin':ymin,
                    'ymax':ymax,
            }
            bboxes.append(single_box)
    return bboxes

def add_bounding_boxes(img_id, sample_img, annotation_dir):
    all_boxes = getBBox(annotation_dir, img_id)
    for bb in all_boxes:
        image = cv2.rectangle(sample_img, (bb['xmin'], bb['ymin']), (bb['xmax'], bb['ymax']), (255, 255, 255), 5)
    return image

class VisEnsembleFeature():

    def __init__(self, obj_dir=None, annotation_dir=None,
                img_run_path=None, vis_dir=None) :

        self.obj_dir = obj_dir
        self.img_run_path = img_run_path
        self.vis_dir = vis_dir
        self.annotation_dir = annotation_dir
        self.img_id = self.img_run_path.split('/')[-1]
        
    def compute_avg_fv_imgs(self):
        sample_img_path = os.path.join(self.obj_dir, self.img_id + '.jpg')
        sample_img = cv2.imread(sample_img_path)
        sample_img = add_bounding_boxes(self.img_id, sample_img, self.annotation_dir)
        

        pixels_information = []
        all_fv_imgs = []
        for json_file_name in os.listdir(self.img_run_path):

            if '.json' in json_file_name:
                json_file_path = os.path.join(self.img_run_path, json_file_name)
                with open(json_file_path) as json_file: result_data = json.load(json_file)
                
                # load feature map result from runs file
                pxl_lvl_anom_score = torch.tensor(result_data['pixel_score'])
                score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
                fv_img = pred_to_img(pxl_lvl_anom_score, score_range)

                pixels_information.append(result_data['count_pixels'])
                all_fv_imgs.append(fv_img)
                
        fv = avg_by_small(pixels_information, all_fv_imgs)
        return fv, sample_img
    
    def save_opt_fv(self):
        fv, sample_img= self.compute_avg_fv_imgs()
        plt.imshow(sample_img)
        plt.imshow(fv, cmap="jet", alpha=0.5)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        overlay_img = Image.open(buf)
        plt.clf()

        save_img_path = os.path.join(self.vis_dir, self.img_id + '.png')
        overlay_img.save(save_img_path)


if __name__ == "__main__":

    run_dir = '/media/rc/backup/exp/2022_07_21_21_02_52/runs'
    vis_dir = '/media/rc/backup/exp/2022_07_21_21_02_52/vis'
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)

    for img_run_dir in os.listdir(run_dir):
        img_run_path = os.path.join(run_dir, img_run_dir)
        if os.path.isdir(img_run_path):
            vis_exp = VisEnsembleFeature(obj_dir='./datasets/full_body/test/objs', 
                                        annotation_dir='./datasets/full_body/Annotations',
                                        img_run_path=img_run_path, 
                                        vis_dir=vis_dir)
            vis_exp.save_opt_fv()



        