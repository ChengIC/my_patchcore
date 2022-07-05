
from config_utils import *
from train_utils import *
from inference_utils import *

class TwoStageCore():

    def __init__(self, 
                exp_dir = './TwoStageCore/exp',
                normal_img_dir=None, 
                obj_img_dir=None) :

        self.exp_dir=exp_dir
        self.normal_img_dir=normal_img_dir
        self.obj_img_dir=obj_img_dir
    
    def getPatchesImage (self, run_dir, img_dir, th=0, save_folder_name='cuts_1'):

        cutDir = os.path.join ('/'.join(self.FirstConifgDir.split('/')[:-1]), save_folder_name)
        if not os.path.exists(cutDir):
            os.makedirs(cutDir)

        for file in tqdm(os.listdir(run_dir)):
             if '.json' in file:
                
                img_id = file.split('_config')[0]
                config_id = file.split('config_')[1].split('.json')[0]

                img_path = os.path.join(img_dir, img_id + '.jpg')
                img = cv2.imread(img_path)

                with open(os.path.join(run_dir,file)) as run_file:
                    run_data = json.load(run_file)
                    detected_box_list = run_data['detected_box_list']
                    for selected_th in detected_box_list:
                        if float(selected_th) > th:
                            detected_box = detected_box_list[selected_th]
                            for bb in detected_box:
                                crop = img[bb[1]:bb[3], bb[0]:bb[2]]
                                crop_img_name = '{}_config_{}_crop_{}.jpg'.format(img_id, config_id, unique_id(12))
                                
                                save_crop_path = os.path.join(cutDir, crop_img_name)
                                cv2.imwrite(save_crop_path, crop)

                                exp_info ={
                                        'img_id':img_id,
                                        'save_crop_path':crop_img_name,
                                        'bbox':bb,
                                        'threshold':selected_th,
                                }

                                json_string = json.dumps(exp_info)
                                crop_json_name = crop_img_name.replace('jpg','json')
                                crop_json_path = os.path.join(cutDir, crop_json_name)
                                with open(crop_json_path, 'w') as outfile:
                                    outfile.write(json_string)

        print('finish cropping and return cut dir')
        return cutDir

    def train (self):

        # train first patch-core
        self.FirstConifgDir = GenConfigureFiles(self.normal_img_dir, 
                                                self.exp_dir,
                                                save_name='config_1').genMultiScaleFiles(method='random', bacth_nums=1, 
                                                                                        num_imgs_list = [70],
                                                                                        scale_list=[0.1])
        self.FirstModelDir = TrainPatchCore(self.FirstConifgDir, save_name='models_1').trainModel()
        self.FirstRunDir =  InferenceCore(self.FirstModelDir,save_name='runs_1').inference_one_model(self.normal_img_dir)
        self.FirstCutDir = self.getPatchesImage(run_dir=self.FirstRunDir,
                                                img_dir=self.normal_img_dir,
                                                save_folder_name='cuts_1')

        # train second patch-core
        self.SecondConifgDir = GenConfigureFiles(self.FirstCutDir, self.exp_dir, save_name='config_2',
                                config_folder=os.path.join ('/'.join(self.FirstConifgDir.split('/')[:-1]))).genMultiScaleFiles(method='random', 
                                                                                                                                bacth_nums=1, 
                                                                                                                                num_imgs_list = [100],
                                                                                                                                scale_list=[0.5])
        self.SecondModelDir = TrainPatchCore(self.SecondConifgDir, save_name='models_2').trainModel()


    def inference(self):
        self.SecondRunDir =  InferenceCore(self.FirstModelDir,save_name='inference_runs_on_img').inference_one_model(self.obj_img_dir)

        self.SecondCutDir = self.getPatchesImage(run_dir=self.SecondRunDir,
                                                    img_dir=self.obj_img_dir,
                                                    th = 0.7,
                                                    save_folder_name='cut_abnormal_patch')

        self.FinalRunDir = InferenceCore(self.SecondModelDir,save_name='final_runs').inference_one_model(self.SecondCutDir)


if __name__ == "__main__":
    normal_img_dir = './datasets/full_body/train/good'
    obj_img_dir = './datasets/full_body/test/objs'

    tS = TwoStageCore(normal_img_dir=normal_img_dir,obj_img_dir=obj_img_dir)
    tS.train()
    tS.inference()