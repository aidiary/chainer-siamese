import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


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
        loss = F.contrastive(x0, x1, label, margin=1)

        reporter.report({'loss': loss}, self)

        return loss


class SiameseNetwork_v2(chainer.Chain):

    def __init__(self):
        super(SiameseNetwork_v2, self).__init__(
            conv1=L.Convolution2D(None, 6, ksize=3, pad=1),
            conv2=L.Convolution2D(None, 16, ksize=3, pad=1),
            conv3=L.Convolution2D(None, 16, ksize=3, pad=1),
            conv4=L.Convolution2D(None, 16, ksize=3, pad=1),
            conv5=L.Convolution2D(None, 16, ksize=3, pad=1),
            conv6=L.Convolution2D(None, 16, ksize=3, pad=1),

            bn1=L.BatchNormalization(6),
            bn2=L.BatchNormalization(16),
            bn3=L.BatchNormalization(16),
            bn4=L.BatchNormalization(16),
            bn5=L.BatchNormalization(16),
            bn6=L.BatchNormalization(16),

            fc1=L.Linear(None, 120),
            fc2=L.Linear(None, 84),
            fc3=L.Linear(None, 10)
        )
        self.train = True

    def forward_once(self, x_data):
        h = F.relu(self.bn1(self.conv1(x_data), test=not self.train))
        h = F.relu(self.bn2(self.conv2(h), test=not self.train))
        h = F.max_pooling_2d(h, ksize=2)

        # conv block 2
        h = F.relu(self.bn3(self.conv3(h), test=not self.train))
        h = F.relu(self.bn4(self.conv4(h), test=not self.train))
        h = F.max_pooling_2d(h, ksize=2)

        # conv block 3
        h = F.relu(self.bn5(self.conv5(h), test=not self.train))
        h = F.relu(self.bn6(self.conv6(h), test=not self.train))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)

        return y


    def __call__(self, x0_data, x1_data, label):
        # 双子で同じCNNを使うことでパラメータは共有する
        x0 = self.forward_once(x0_data)
        x1 = self.forward_once(x1_data)

        # contrastive lossを返す
        loss = F.contrastive(x0, x1, label, margin=1)

        reporter.report({'loss': loss}, self)

        return loss
