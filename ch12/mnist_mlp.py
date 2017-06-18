import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte'%kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte'%kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num , row, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('mnist',kind='train')
print('Rows: %d, columnsL %d' %(X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columnsL %d' %(X_test.shape[0], X_test.shape[1]))



# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train==i][0].reshape(28,28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
#
#
# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(25):
#     img = X_train[y_train==7][i].reshape(28,28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
#
# np.savetxt('train_img.csv', X_train,fmt='%i', delimiter='.')
# np.savetxt('train_labels.csv', y_train,fmt='%i', delimiter='.')
# np.savetxt('test_img.csv', X_test,fmt='%i', delimiter='.')
# np.savetxt('test_labels.csv', y_test,fmt='%i', delimiter='.')

nn = NeuralNetMLP(n_output=10,
                  n_features = X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  shuffle=True,
                  minibatches=50,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

plt.plot(range(len(nn.cost_)),nn_cost_)
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()