# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm  
import os
import json
import torchvision
from torchvision import transforms
import torch.optim as optim
from utils_for_video import read_split_data
from my_dataset import MyDataSet
from matplotlib import pyplot as plt
import numpy as np
    
    
class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet152(pretrained = True)
        layers = list(self.resnet.children())
        
        self.layer = nn.Sequential(*layers[:8])
        self.out = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.classifier = nn.Linear(2048, num_classes)

                                        
    def forward(self, x):      
        x = self.layer(x)
        x = self.out(x)
        p = self.classifier(x)
        
        return p
    

def change_lr(net, lr, gamma = 0.8):
    if lr >= 0.0000001:
        lr = gamma * lr
    return lr, optim.Adam(net.parameters(), lr = lr, weight_decay = 0.0001)

def visualize(data: list):
    plt.figure(figsize=(20, 8), dpi=80)

    a_x = range(len(data))
    a_y = data
    ary = [1, 2]

    # 使用scatter方法绘制散点图
    plt.scatter(a_x, a_y)

    plt.xticks(range(0,350, 7))
    plt.yticks(np.linspace(0,1, 50))
    plt.xlabel('number',fontsize=14)
    plt.ylabel('score',fontsize=14)
    plt.title('false prediction scores',fontsize=24)

    plt.grid(alpha = 0.4)

    #plt.savefig('气温散点图')

    plt.show()
    
        

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
        "val": transforms.Compose([transforms.Resize(224),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = "./data"

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("./UCF-101-frame")

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    batch_size = 20
    nw = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)


    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model_name = "single_frame_selection_video_152"
    net = ResNet().to(device)
    model_weight_path = "./single_frame_selection_video_152.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
      
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch   
    with torch.no_grad():
        
        train_bar = tqdm(train_loader)
        for train_data in train_bar:
            train_images, train_labels = train_data
            outputs = net(train_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, train_labels.to(device)).sum().item()
        train_accurate = acc / train_num
        
        acc = 0
        t, num_over_t = 0.5, 0
        num_list, num_false_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        num = 0
        val_bar = tqdm(validate_loader)
        false_list = []
        sigmoid = nn.Sigmoid()
        
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = sigmoid(net(val_images.to(device)))
            predict_y = torch.max(outputs, dim=1)[1]
            bool_list = torch.eq(predict_y, val_labels.to(device))
            for i in range(len(val_labels)):
                num_list[val_labels[i].item()] += 1
            for i in range(len(bool_list)):
                if bool_list[i] == False:
                    #print(bool_list)
                    #print(outputs)
                    #print(val_labels)
                    false_pre = outputs[i][val_labels[i]].item()
                    num_false_list[val_labels[i].item()] += 1
                    if false_pre < 0.2:
                        print(val_images_path[num + i])
                    #if false_pre > 0.95:
                        #print(bool_list)
                        #print(outputs)
                        #print(val_labels)
                        #assert 0
                    #print(false_pre)
                    #assert 0
                    if false_pre > t:
                        num_over_t += 1
                    false_list.append(round(false_pre, 4))
            num += 20
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            
        val_accurate = acc / val_num
    #print('train_accuracy: %.4f' % (train_accurate))
    print('val_accuracy: %.4f' % (val_accurate))
    print(f"number of false prediction is {len(false_list)},the min number is {min(false_list)}, the max number is {max(false_list)}, the number of false prediction over t is {round(num_over_t/len(false_list), 4)}")
    print(num_list, sum(num_list))
    print(num_false_list)
    print(num)
    #visualize(false_list)

    assert 0
    
    
    loss_function = nn.CrossEntropyLoss()

    epochs = 100
    best_acc = 0.8882
    save_path = './{}.pth'.format(model_name)
    train_steps = len(train_loader)   

    
    lr = 0.0003
    jsq = 0
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = 0.0001)
                       
    for epoch in range(epochs):
        # train
        net.train()
        if(jsq == 8):
            jsq = 0
            lr, optimizer = change_lr(net, lr)
            print(f"the lr from epoch{epoch} is {lr}")
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            
            p = net(images.to(device))
            loss = loss_function(p, labels.to(device))
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.4f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            jsq = 0
            torch.save(net.state_dict(), save_path)
            print("  Parameters have been stored")
            
        else:
            jsq += 1
            

    print('Finished Training')

    
#用于给所有的frame打分时读取数据
def score(root):
    debug = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    transform = transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
      
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("./UCF-101-frame")

    # 实例化验证数据集
    dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transform)
    
    batch_size = 15
    nw = 0
    data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    
    img_num = len(dataset)
    print(f"there are {img_num} frames will be scored")

    net = ResNet().to(device)
    model_weight_path = "./single_frame_selection_video_152.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    acc = 0.0  # accumulate accurate number / epoch  
    sigmoid = nn.Sigmoid()
    scores=[]
    net.eval()
    with torch.no_grad():
        bar = tqdm(data_loader)
        for data in bar:
            images, labels = data
            outputs = sigmoid(net(images.to(device)))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
            i = 0
            for pre in outputs:  
                score = pre[labels[i]]
                scores.append(round(score.item(), 5))#保留小数点后5位
                i += 1
        accurate = acc / img_num
    print('accuracy on these frames: %.4f' % (accurate))
    #scroces是一维的，因为没有打乱顺序，所以顺序上是一一对应的
    return scores
    

if __name__ == '__main__':
    #是否预训练
    pretrain = False
    if pretrain:
        main()
    else:
        scores = score("./UCF-101-frame")