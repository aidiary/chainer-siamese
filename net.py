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


    def forward_once(self, x):
        pass


    def __call__(self, x0, x1, label):
        # contrastive lossを返す
        pass
