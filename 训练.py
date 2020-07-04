import tensorflow.python.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# 直接联网加载mnist数据集，也可以先下载好，然后再导入

X_train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
X_test4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

X_train4D_Normalize = X_train4D / 255 # 归一化
X_test4D_Normalize = X_test4D / 255

y_trainOnehot = to_categorical(y_train)
y_testOnehot = to_categorical(y_test)
# Sequential 用于建立序列模型
# Flatten 层用于展开张量，input_shape 定义输入形状为 28x28 的图像，展开后为 28*28 的张量。
# Dense 层为全连接层，输出有 128 个神经元，激活函数使用 relu。
# Dropout 层使用 0.25 的失活率。
# 再接一个全连接层，激活函数使用 softmax，得到对各个类别预测的概率。

model = Sequential()

# 一层卷积
model.add(
    Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='same',  # 保证卷积核大小，不够补零
        input_shape=(28, 28, 1),
        activation='relu'))
# 池化层1
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 二层卷积
model.add(
    Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
# 池化层2
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(
    Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # 平坦层
model.add(Dense(128, activation='relu'))  # 全连接层
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax')) # 激活函数

# 优化器选择 Adam 优化器。
# 损失函数使用 categorical_crossentropy，
# sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。

# 训练模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(x=X_train4D_Normalize,
                          y=y_trainOnehot,
                          validation_split=0.2,
                          batch_size=300,
                          epochs=10,
                          verbose=2)


# fit 用于训练模型，对训练数据遍历一次为一个 epoch，这里遍历 10 次。
# evaluate 用于评估模型，返回的数值分别是损失和指标。
# model.fit(x_train,y_train,epochs=10)
# 将整个模型保存为HDF5文件
model.save('D:/my_model.h5')
model.evaluate(X_test4D_Normalize, y_testOnehot)