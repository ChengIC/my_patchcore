
import torch
from PIL import Image
from models import PaDiM, PatchCore,SPADE
from torchvision import transforms
from torch import tensor
import matplotlib.pyplot as plt
import io
import streamlit as st
import os 
import time
import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])   

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x, range): 
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)

def show_pred(sample, score, fmap, range):
    sample_img = tensor_to_img(sample, normalize=True)
    fmap_img = pred_to_img(fmap, range)
    # overlay
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    # buf.seek(0)
    overlay_img = Image.open(buf)

    return overlay_img

loader=transforms.Compose([
                # transforms.Resize([224,224], interpolation=transforms.InterpolationMode.BICUBIC),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

model = PatchCore(
                f_coreset=0.1, 
                backbone_name="wide_resnet50_2",
             )

model.load_state_dict(torch.load('./model_zoo/patchcore_ratio_10/patchcore_path'))
var1, var2, var3, var4, var5, var6 = torch.load('./model_zoo/patchcore_ratio_10/patchcore_path.tar')

config_setting = 'patchcore_ratio_10'
my_folder = './datasets/full_body/test/objs'
output_folder = './output_imgs/'
now = int(time.time())
timeArray = time.localtime(now)
StyleTime = time.strftime("%Y_%m_%d %H_%M_%S_", timeArray)
output_folder = output_folder + config_setting + '/' + StyleTime + '/'
if not os.path.exists(output_folder):
    os.makedirs (output_folder)


for image_name in os.listdir(my_folder):
    if 'jpg'in image_name:
        image_path = os.path.join(my_folder,image_name)
        image = Image.open(image_path).convert('RGB')
        image = loader(image).unsqueeze(0)
        test_img_tensor = image.to('cpu', torch.float)

        # PatchCore
        img_lvl_anom_score, pxl_lvl_anom_score = model.inference(test_img_tensor, var1, var2, var3, var4, [880,335], var6 )
        # img_lvl_anom_score, pxl_lvl_anom_score = model.inference(test_img_tensor, var1, var2, var3, var4, var5)
        score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
        overlay_img = show_pred(test_img_tensor.squeeze(0), img_lvl_anom_score, pxl_lvl_anom_score, score_range)

        output_img_path = output_folder + str(img_lvl_anom_score.numpy()) +'_'+ image_name.replace('.jpg','_output.png')

        img = Image.open(image_path).convert('RGB')
        img_size = img.resize((200, 200))
        img1_size = overlay_img.resize((200, 200))
        img2 = Image.new("RGB", (400, 200), "white")
        img2.paste(img_size, (0, 0))
        img2.paste(img1_size, (200, 0))
        img2.save(output_img_path)

