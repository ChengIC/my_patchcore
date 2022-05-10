
# model_dict = models_dict[single_model]
from cores_utils import *
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

def inference_single_core(model_dict,img_path,run_dir):
	roots = model_dict['roots']
	model = model_dict['model']
	model_paras = model_dict['model_paras']
	resize_box = model_dict['resize_box']

	# load image tensor
	loader = loader_from_resize(resize_box)
	image = Image.open(img_path).convert('RGB')
	original_size_width, original_size_height = image.size
	image = loader(image).unsqueeze(0)
	test_img_tensor = image.to('cpu', torch.float)

	# inference
	HeatMap_Size = [original_size_height, original_size_width]
	_, pxl_lvl_anom_score = model.inference (test_img_tensor, model_paras, HeatMap_Size)
	detected_box_list = PixelScore2Boxes(pxl_lvl_anom_score)

	# log exp
	img_id = img_path.split('/')[-1].split('.')[0]
	s = roots.split('/')[-1]
	json_file_name = img_id + '_config_' + s + '.json'
	json_filePath = os.path.join(run_dir, json_file_name)

	exp_info ={
		'img_path':img_path,
		'detected_box_list':detected_box_list,
		'resize_box':resize_box,
		'roots_dir':roots,
		'img_id':img_id,
		'json_id':s
	}

	json_string = json.dumps(exp_info)
	with open(json_filePath, 'w') as outfile:
		outfile.write(json_string)

	return exp_info