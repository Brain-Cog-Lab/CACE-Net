import os

import h5py
import torch
from tqdm import tqdm
import torch
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import torchvision.transforms as transforms
import certifi
import ssl
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import argparse

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--ckpt",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
args = parser.parse_args()

def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num


if torch.cuda.is_available():
    device = torch.device('cuda')

# 加载预训练的 ResNet50 模型
weights = ResNet50_Weights.IMAGENET1K_V2

# 获取针对ResNet50的推理预处理操作
preprocess = weights.transforms()


# 文件夹路径
folder_path = '/home/hexiang/Encoders/data/AVE_Dataset/AVE/'

lis = []
# 打开并读取文件
with open('/home/hexiang/Encoders/data/AVE_Dataset/Annotations.txt', 'r') as file:
    for line in file:
        # 分割每一行，并获取文件名（即第二部分）
        parts = line.strip().split('&')
        if len(parts) > 1:  # 确保有足够的部分来提取文件名
            file_name = parts[1]
            lis.append(file_name + '.mp4')

video_features = torch.zeros([len(lis), 10, 2048, 7, 7])  # 10s long video
t = 10 # length of video
sample_num = 16 # frame number for each second


# 假设预处理函数已经定义
def preprocess_frame(frame, preprocess):
    x_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
    x_im = Image.fromarray(x_im)  # 将numpy数组转换为PIL图像
    x_im = preprocess(x_im)  # 应用预处理
    return x_im


preprocess_frame_partial = partial(preprocess_frame, preprocess=preprocess)

# 遍历文件夹中的所有 .h5 文件
def create_h5(filename):
    '''feature learning by VGG-net'''
    video_index = os.path.join(folder_path, filename)  # path of videos

    cap = cv2.VideoCapture(video_index)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(vid_len / t)

    frame_num = video_frame_sample(frame_interval, t, sample_num)

    imgs = []
    while True:
        ret, pre_frame = cap.read()
        if not ret:
            break
        imgs.append(pre_frame)

    res_imgs = []

    with ThreadPoolExecutor(max_workers=16) as exe:
        # res_imgs = list(tqdm(exe.map(preprocess_frame_partial, imgs), total=len(imgs)))
        res_imgs = list(exe.map(preprocess_frame_partial, imgs))

    # extract_frame = torch.zeros((160, 3, 224, 224))  # 10 seconds * 16 frames per second  # 这里不知道为什么会引起多线程报错


    # extract_frame[i] = res_imgs[n] 也会报错.. 只能改成列表形式了; 列表中的元素是numpy因为h5只能保存numpy而非tensor.
    extract_frame = []
    idx = torch.arange(0, 160, 16) + torch.randint(0, 16, (10,))
    for i, n in enumerate(frame_num):
        if i in idx:
            extract_frame.append(res_imgs[n].numpy())

    output_filename = os.path.join("/home/hexiang/Encoders/data/image_opencv_finetune/", filename[:-4] + '.h5')
    with h5py.File(output_filename, 'w') as hf:
        # 使用压缩来保存extract_frame数组
        hf.create_dataset('extract_frame', data=extract_frame)

    cap.release()

if __name__ == '__main__':
    frame_save = False

    if frame_save:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []

            for filename in lis:
                future = executor.submit(create_h5, filename)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    else:
        # 文件夹路径
        folder_path = '/home/hexiang/Encoders/data/image_opencv'

        lis = []
        # 打开并读取文件
        with open('/home/hexiang/Encoders/data/AVE_Dataset/Annotations.txt', 'r') as file:
            for line in file:
                # 分割每一行，并获取文件名（即第二部分）
                parts = line.strip().split('&')
                if len(parts) > 1:  # 确保有足够的部分来提取文件名
                    file_name = parts[1]
                    lis.append(file_name + '.h5')

        video_features = torch.zeros([len(lis), 10, 2048, 7, 7])  # 10s long video

        # define model
        model = resnet50(weights=weights, progress=True)
        model.eval()
        model = model.to(device)
        model = torch.nn.Sequential(*(list(model.children())[:-2]))

        load_path = os.path.join("/home/hexiang/AVELCLIP/audioset_tagging_cnn/", args.ckpt)
        print("load ckpt from:{}".format(load_path))
        model.load_state_dict(torch.load(load_path, map_location=device)["I_state_dict"])

        # inference
        for i, filename in enumerate(tqdm(lis)):
            if filename.endswith('.h5'):
                file_path = os.path.join(folder_path, filename)

                # 读取 .h5 文件
                with h5py.File(file_path, 'r') as h5_file:
                    # 假设你想从每个文件中读取名为 'data' 的数据集
                    # 你需要根据你的文件结构进行调整
                    data = h5_file['extract_frame'][:]
                    # 将数据转换为 PyTorch 张量
                    tensor = torch.tensor(data)

            tensor = tensor.to(device)
            with torch.no_grad():
                pool_feature = model(tensor)
            pool_feature = pool_feature.reshape((10, 16, 2048, 7, 7))
            feature_vector = torch.mean(pool_feature, dim=1)
            video_features[i, :, :, :, :] = feature_vector

        video_features = video_features.permute(0, 1, 3, 4, 2)

        with h5py.File('/home/hexiang/Encoders/data/visual_ckpt_{}_feature.h5'.format(args.ckpt.split('/')[-1].split('.')[0]), 'w') as hf:
            hf.create_dataset("dataset", data=video_features)