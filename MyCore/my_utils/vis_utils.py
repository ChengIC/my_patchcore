from my_utils.inference_utils import *
import matplotlib.pyplot as plt
import io
from PIL import Image




def PixelScore2Img(pixel_score):
    pxl_lvl_anom_score = torch.tensor(pixel_score)
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    fv_img = pred_to_img(pxl_lvl_anom_score, score_range)

    plt.imshow(fv_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    fv_color_img = Image.open(buf)
    plt.clf()
    return fv_color_img