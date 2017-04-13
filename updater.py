import chainer
import chainer.functions as F
from chainer import Variable


class SiameseUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.siamese = kwargs.pop('model')
        super(SiameseUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        # xのペアデータを作成（正例・負例のバランスを取る）

        # 同じクラスか異なるクラスかのラベルを作成

        # SiameseNetworkの出力（contrastive loss）を計算
        # モデルの
        # optimizerを更新

        pass
