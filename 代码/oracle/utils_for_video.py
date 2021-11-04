import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

from matplotlib import pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    debug = 1
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    action_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    action_class.sort()

    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(action_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    all_val = []
    for cla in action_class:
        cla_path = os.path.join(root, cla)#某一类的video
        video_name = [name for name in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, name))]#video文件的名字, list
        videos = [os.path.join(cla_path, name) for name in video_name]#某类下每个video的地址, list
    
        # 记录该类别的视频数量
        every_class_num.append(len(videos))
        # 按比例随机采样验证样本
        val_path = random.sample(videos, k=int(len(videos) * val_rate))

        # 获取该类别对应的索引
        video_class = class_indices[cla]
        for video_path in videos:
            if video_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                # 遍历获取supported支持的所有文件路径，所有video下的图片地址
                images = [os.path.join(video_path, i) for i in os.listdir(video_path)
                          if os.path.splitext(i)[-1] in supported]
                for img_path in images:
                    val_images_path.append(img_path)
                    val_images_label.append(video_class)
            else:  # 否则存入训练集
                images = [os.path.join(video_path, i) for i in os.listdir(video_path)
                           if os.path.splitext(i)[-1] in supported]
                for img_path in images:
                    train_images_path.append(img_path)
                    train_images_label.append(video_class) 
     
    print("{} videos were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(action_class)), every_class_num)
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(action_class)), action_class, rotation = 45)
        # 在柱状图上添加网格
        plt.grid(alpha = 0.4)
        # 设置x坐标
        plt.xlabel('video class')
        # 设置y坐标
        plt.ylabel('number of videos')
        # 设置柱状图的标题
        plt.title('video class distribution')
        plt.show()  

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_data(root:str):
    #oracle对帧进行打分时，读取文件的函数，此时不需要划分测试集与数据集
    debug = 1
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    action_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    action_class.sort()

    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(action_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices_score.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  # 存储训练集的所有图片路径
    images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    
    for cla in action_class:
        cla_path = os.path.join(root, cla)#某一类的video
        video_name = [name for name in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, name))]#video文件的名字, list
        videos = [os.path.join(cla_path, name) for name in video_name]#某类下每个video的地址, list

        video_class = class_indices[cla]
        num = 0
        for video_path in videos:
            images = [os.path.join(video_path, i) for i in os.listdir(video_path)
                        if os.path.splitext(i)[-1] in supported]
            for img_path in images:
                images_path.append(img_path)
                images_label.append(video_class)
                num+=1
        every_class_num.append(num)
        
    return images_path, images_label, every_class_num                
                    
          
