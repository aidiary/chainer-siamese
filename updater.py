import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np


class SiameseUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.siamese = kwargs.pop('model')
        super(SiameseUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        # xのペアデータを作成
        # TODO: 正例・負例のバランスは？
        batch = self.get_iterator('main').next()

        # converter()を通すとconcat_examples()によりサンプルがマージされたarrayになる
        train_data, train_label = self.converter(batch, self.device)

        print(train_data.shape)   # (num_batch, 1, 28, 28)
        print(train_label.shape)  # (num_batch, )

        exit()

        y = self.siamese(train_data, train_data, train_label == train_label)
        optimizer = self.get_optimizer('main')

        exit()

        # 同じクラスか異なるクラスかのラベルを作成

        # SiameseNetworkの出力（contrastive loss）を計算
        # optimizerを更新
