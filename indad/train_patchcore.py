from train_utils import LoadTrainConfig
from models import PatchCore
from save_utils import saveModelPath
import torch
import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")


class train_patchcore():
    def __init__(self,configPath,train_imgs_folder, 
                f_coreset=.20,backbone_name="wide_resnet50_2",TimeStamp=None):
        
        self.model=PatchCore(
                    f_coreset=f_coreset, 
                    backbone_name=backbone_name,
                )
        self.configPath=configPath
        self.train_imgs_folder=train_imgs_folder
        self.TimeStamp=TimeStamp

    def run(self): 
        train_loading = LoadTrainConfig(self.configPath)
        train_ds = train_loading.genTrainDS(self.train_imgs_folder)
        train_tar, train_path = saveModelPath(self.configPath,self.TimeStamp)
       
        tobesaved = self.model.fit(train_ds)
        
        torch.save(tobesaved, train_tar)
        torch.save(self.model.state_dict(), train_path)

        return train_tar, train_path

# if __name__ == "__main__":
#     configPath = './config/semi/percentage_0.2_XYWQMHA8.json'
#     train_imgs_folder = './datasets/THz_Body/train/good'
#     my_training = train_patchcore(configPath,train_imgs_folder)
#     train_tar, train_path = my_training.run()
#     print (train_tar, train_path)