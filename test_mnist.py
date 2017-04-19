import os

import chainer
import chainer.links as L
from net import SiameseNetwork

import numpy as np
import matplotlib.pyplot as plt

# 訓練済みモデルをロード
model = SiameseNetwork()
chainer.serializers.load_npz(os.path.join('result', 'model.npz'), model)

# テストデータをロード
_, test = chainer.datasets.get_mnist(ndim=3)
test_data, test_label = test._datasets

# テストデータを学習した低次元空間（2次元）に写像
y = model.forward_once(test_data)
feat = y.data

# ラベルごとに描画
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']

# 各ラベルごとに異なる色でプロット
# 同じクラス内のインスタンスが近くに集まり、
# 異なるクラスのインスタンスが離れていれば成功
for i in range(10):
    f = feat[np.where(test_label == i)]
    plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.savefig(os.path.join('result', 'result.png'))
