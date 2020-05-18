import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Input, Dense,Activation
from keras.optimizers import RMSprop
from keras.layers.merge import _Merge
from keras import backend as K
from functools import partial
from sklearn.metrics import mean_squared_error

class GAN:
    MAX_SIM = 995  # 10000
    NUM_SAMPLES = 995  # 5000
    BATCH_SIZE = 64
    TRAINING_RATIO = 5
    INPUT_DIM = 41
    noise_dim = INPUT_DIM
    def __init__(self, train_df, GRADIENT_PENALTY_WEIGHT, MAX_EPOCH):
        self.train_df = train_df
        self.GRADIENT_PENALTY_WEIGHT = GRADIENT_PENALTY_WEIGHT
        self.MAX_EPOCH = MAX_EPOCH

    def wasserstein_loss(self, y_true, y_pred):
        """ Wasserstein distance Wasserstein距离"""
        return K.mean(y_true * y_pred)

    class RandomWeightedAverage(_Merge):
        def _merge_function(self, inputs):
            weights = K.random_uniform((GAN.BATCH_SIZE, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, lamba_reg):
        """ 计算GP-WGAN的梯度损失"""
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = lamba_reg * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def calcrmse(self, X_train: pd.DataFrame, gensamples: pd.DataFrame):
        """计算均方误差"""
        max_column = X_train.shape[1]
        rmse_lst = []
        for col in range(max_column):
            rmse_lst.append(np.sqrt(mean_squared_error(X_train[:, col], gensamples[:, col])))
        return np.sum(rmse_lst) / max_column

    def generate_samples(self,generator_model, noise_dim, num_samples):
        """生成样本以供进一步分析"""
        return generator_model.predict(np.random.rand(num_samples, noise_dim))

    def writetocsv(self, mtrx, flnm):
        """Save the samples for TDA with R (2nd notebook). We do not differentiate frauds from normal transactions
            保存生成的样本"""
        dtfrm = pd.DataFrame(mtrx)
        dtfrm.to_csv(flnm, sep=',', index=None, header=None)

    def make_generator(self, noise_dim=100):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='he_normal', input_dim = self.INPUT_DIM))  # 输入层维度为INPUT_DIM，第一个隐藏层为128个神经元
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal'))  # kernel_initializer='he_normal'为权重初始化
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(units=noise_dim, activation='linear'))  # 最后一个隐藏层神经元数目为noise_dim，激活函数为线性函数
        return model

    def make_discriminator(self):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='he_normal', input_dim=self.INPUT_DIM))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal', input_dim=self.INPUT_DIM))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal', input_dim=self.INPUT_DIM))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(64, kernel_initializer='he_normal', input_dim=self.INPUT_DIM))
        model.add(Activation('relu'))  # model.add(Activation('relu'))
        model.add(Dense(units=1, activation='linear'))
        return model

    def compile(self):
        generator = self.make_generator(self.noise_dim)
        discriminator = self.make_discriminator()  # 创建生成器和鉴别器

        for layer in discriminator.layers:
            layer.trainable = False
        discriminator.trainable = False  # 固定鉴别器，训练模型之前，需要配置学习过程(compile)，这个过程叫模型编译

        generator_input = Input(shape=(self.noise_dim,))  # 输入层：必须是InputLayer或者Input创建的Tensor
        generator_layers = generator(generator_input)
        discriminator_layers_for_generator = discriminator(generator_layers)

        generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
        generator_model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6),
                                loss=self.wasserstein_loss)  # 配置优化函数，损失函数

        for layer in discriminator.layers:  # 固定生成器
            layer.trainable = True
        for layer in generator.layers:
            layer.trainable = False
        discriminator.trainable = True
        generator.trainable = False

        real_samples = Input(shape=self.train_df.shape[1:])  # 真实数据张量维度
        generator_input_for_discriminator = Input(shape=(self.noise_dim,))
        generated_samples_for_discriminator = generator(generator_input_for_discriminator)
        discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = discriminator(real_samples)

        averaged_samples = self.RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
        averaged_samples_out = discriminator(averaged_samples)

        discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])

        ### the loss function takes more inputs than the standard y_true and y_pred
        ### values usually required for a loss function. Therefore, we will make it partial.
        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=averaged_samples,
                                  lamba_reg=self.GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gp_loss'

        # finally, we compile the model
        discriminator_model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6),
                                    loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss])

        positive_y = np.ones((self.BATCH_SIZE, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((self.BATCH_SIZE, 1), dtype=np.float32)  # 批处理函数上的列必须使用虚拟向量

        for epoch in range(self.MAX_EPOCH + 1):
            np.random.shuffle(self.train_df)

            minibatches_size = self.BATCH_SIZE * self.TRAINING_RATIO
            for i in range(int(self.train_df.shape[0] // (self.BATCH_SIZE * self.TRAINING_RATIO))):
                discriminator_minibatches = self.train_df[i * minibatches_size:(i + 1) * minibatches_size]
                for j in range(self.TRAINING_RATIO):
                    sample_batch = discriminator_minibatches[j * self.BATCH_SIZE:(j + 1) * self.BATCH_SIZE]
                    noise = np.random.rand(self.BATCH_SIZE, self.noise_dim).astype(np.float32)

                    discriminator_model.train_on_batch([sample_batch, noise], [positive_y, negative_y, dummy_y])

                generator_model.train_on_batch(np.random.rand(self.BATCH_SIZE, self.noise_dim), positive_y)

            if (epoch % 1000 == 0):
                gensamples = self.generate_samples(generator, self.noise_dim, self.MAX_SIM)
                rmse_sofar = self.calcrmse(self.train_df, gensamples)  # 均方根误差，rmse反映了测量数据偏离真实值的程度，rmse越小，表示测量精度越高。
                print("Epoch: ", epoch, "\t", "rmse: ", rmse_sofar)

        generatorgpwgan = generator
        generated = self.generate_samples(generatorgpwgan, self.noise_dim,
                                          self.NUM_SAMPLES)  # 在generator中，predict()输出的数据便是生成的Attack traffic
        print(self.calcrmse(self.train_df, generated))
        self.writetocsv(generated, "./generated_samples/GPWGAN_generated.csv")