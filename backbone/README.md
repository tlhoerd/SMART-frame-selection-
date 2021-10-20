# SMART所使用的backbone网络及其预训练参数

the lightweight features used for frame selection are
computed using MobileNet(Sandler et al. 2018) and
GloVe(Pennington, Socher, and Manning 2014). After the
frame selection is done, we can use a more expensive and
high-quality feature representation. In our experiments, we
use three different backbones: ResNet-152, ResNet-101(He
et al. 2016), and Inception-v3(Szegedy et al. 2017). The
backbones are pre-trained either on ImageNet(Deng et al.
2009) or Kinetics.
