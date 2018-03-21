
import tflearn
import tensorflow as tf
import numpy as np
from tflearn.datasets import mnist
import logging
logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    level=logging.DEBUG)

'''
X:图像数据集,大小是(55000, 784),共55000个图片,每个图片有784维度
Y:图像标签,(55000, 10),表示数字0-9
X_test:(10000, 784)
Y_test:(10000, 10)
'''
X,Y,X_test,Y_test=mnist.load_data(one_hot=True)
logging.info('X = {}'.format(X.shape))
logging.info('Y = {}'.format(Y.shape))
logging.info('X_test = {}'.format(X_test.shape))
logging.info('Y_test = {}'.format(Y_test.shape))


image_dim=X.shape[1]        # 图片维度784
logging.info('image_dim = {}'.format(image_dim))

z_dim=200  # 噪音的大小
#total_samples=X.shape[0]
total_samples=len(X)        #样本大小
logging.info('total_samples = {}'.format(total_samples))


# 生成模型Generater
def generator(x,reuse=False):
    with tf.variable_scope('Generater',reuse=reuse):
        x = tflearn.fully_connected(x, 256,       activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x

# 判别模型Discriminator
def discriminator(x,reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, 1,   activation='sigmoid')
        return x

# 构建网络
gen_input  = tflearn.input_data(shape=[None,z_dim],     name='input_noise')         # (, 200)
disc_input = tflearn.input_data(shape=[None,image_dim], name='disc_input')          # (, 784)

#生成器,判别器
gen_sample = generator(gen_input)
disc_real  = discriminator(disc_input)                  # 判别网络
disc_fake  = discriminator(gen_sample,reuse=True)       # 欺骗网络D


# 定义损失函数
disc_loss = -tf.reduce_mean(tf.log(disc_real)+tf.log(1.-disc_fake))
gen_loss  = -tf.reduce_mean(tf.log(disc_fake))

# 建立训练的优化器并且载入自定义的损失函数
gen_vars  = tflearn.get_layer_variables_by_scope('Generater')
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')

gen_model  = tflearn.regression(gen_sample, placeholder=None, optimizer='adam',
                                loss=gen_loss,  trainable_vars=gen_vars,  batch_size=64,
                                name='target_gen',  op_name='GEN')
disc_model = tflearn.regression(disc_real,  placeholder=None, optimizer='adam',
                                loss=disc_loss, trainable_vars=disc_vars, batch_size=64,
                                name='target_disc', op_name='DISC')

# 训练
gen = tflearn.DNN(gen_model, checkpoint_path='./model/gan/model_gan', tensorboard_dir='./logs')

# 生成模型传入的是噪音，那么我们就需要构建一个噪音数据
z = np.random.uniform(-1., 1., size = [total_samples, z_dim])

# 数据传入进行训练
gen.fit(X_inputs={gen_input: z, disc_input: X}, Y_targets=None, n_epoch=100, run_id='gan_mnist')

# 可视化模型训练效果
import matplotlib.pyplot as plt
f, axis = plt.subplots(10, 2, figsize=(10, 4))
for i in range(10):
    for j in range(2):
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        temp=  [[a,a,a]            for a    in list(gen.predict([z])[0])]
        # temp = [[temp, temp, temp] for temp in list(gan.predict([z])[0])]
        axis[i][j].imshow(np.reshape(temp, (28,28,3)))
f.show()
plt.draw()
