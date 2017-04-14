import chainer
import chainer.functions as F
import chainer.links as L


class SiameseNetwork(chainer.Chain):

    def __init__(self):
        super(SiameseNetwork, self).__init__(
            conv1=L.Convolution2D(None, 20, ksize=5, stride=1),
            conv2=L.Convolution2D(None, 50, ksize=5, stride=1),
            fc3=L.Linear(None, 500),
            fc4=L.Linear(None, 10),
            fc5=L.Linear(None, 2)
        )


    def forward_once(self, x_data):
        h = F.max_pooling_2d(self.conv1(x_data), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y


    def __call__(self, x0_data, x1_data, label):
        # 双子で同じCNNを使うことでパラメータは共有する
        x0 = self.forward_once(x0_data)
        x1 = self.forward_once(x1_data)

        # contrastive lossを返す
        return F.contrastive(x0, x1, label, margin=1)
