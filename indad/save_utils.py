import os
import time


def genTimeStamp():
    now = int(time.time())
    timeArray = time.localtime(now)
    TimeStamp = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)
    return TimeStamp

def saveModelPath(configPath,TimeStamp=None):
    if TimeStamp is None:
        TimeStamp = genTimeStamp()
    config_filename = configPath.split('/')[-1].strip('.json')
    model_path = './model_zoo/' + config_filename + '/' + TimeStamp
    if not os.path.exists(model_path):
        os.makedirs (model_path)
    train_tar = os.path.join(model_path, 'patchcore_path.tar')
    train_path = os.path.join(model_path, 'patchcore_path')
    return train_tar, train_path

def saveResultPath(configPath,TimeStamp=None,output_folder = './results/'):
    config_filename = configPath.split('/')[-1].strip('.json')
    
    if TimeStamp is None:
        TimeStamp = genTimeStamp()

    output_img_folder = os.path.join(output_folder, config_filename + '/' + TimeStamp + '/' + 'imgs')
    output_data_folder = os.path.join(output_folder, config_filename + '/' + TimeStamp + '/' + 'data')
    if not os.path.exists(output_img_folder):
        os.makedirs (output_img_folder)
    if not os.path.exists(output_data_folder):
        os.makedirs (output_data_folder)
    return output_img_folder,output_data_folder

def folderSet():
    TimeStamp = genTimeStamp()  
    output_folder= os.path.join('./processed_result',TimeStamp)
    output_img_folder = os.path.join(output_folder, 'imgs')
    output_data_folder = os.path.join(output_folder, 'data')
    if not os.path.exists(output_img_folder):
        os.makedirs (output_img_folder)
    if not os.path.exists(output_data_folder):
        os.makedirs (output_data_folder)

    return output_img_folder,output_data_folder