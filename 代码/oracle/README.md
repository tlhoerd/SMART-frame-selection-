# 用于监督单帧选择的oracle模型
该模型为ResNet152(oracle_R)以及EfficientNet v2(oracle_E)，使用了预训练权重，目前在单独划分出的UCF101上训练过。

我们在视频维度上来划分训练集与测试集，目前在小数据集上ResNet152正确率为88.2%，EfficienNet v2-s的正确率为94.4%

oracle中的score函数用于给一整个数据集中的所有帧打分
