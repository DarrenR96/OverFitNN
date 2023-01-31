import tensorflow as tf 

class overfitNN(tf.keras.Model):
    def __init__(self, CNNLayers=[16, 32, 32, 64, 64, 128, 256], outputLayers=[256,3], upsamplingType='bicubic'):
        super().__init__()
        self.upSampleLayers = []
        for _layer in CNNLayers:
            self.upSampleLayers.extend([
                cnnOpPadded(_layer),
                tf.keras.layers.UpSampling2D((2,2),interpolation=upsamplingType)
            ])
        self.finalLayers = []
        for _layer in outputLayers[:-1]:
            self.finalLayers.append(cnnOpPadded(_layer))
        self.finalLayers.append(cnnOpPadded(outputLayers[-1], activation='tanh'))

    def call(self, x, training=False):
        x = tf.ones((tf.shape(x)[0], 4, 4, 1)) * tf.reshape(x, (-1,1,1,1))
        for _layer in self.upSampleLayers:
            x = _layer(x, training=training)
        for _layer in self.finalLayers:
            x = _layer(x, training=training)
        x = (x + 1)/2.0
        return x

    def model(self):
        f_t = tf.keras.Input(shape=(1), name='Frame number')
        return tf.keras.Model(inputs=[f_t], outputs=self.call(f_t))


class cnnOpPadded(tf.keras.layers.Layer):
    def __init__(self, filter=32,ksize=5,padding='VALID',activation=tf.keras.layers.LeakyReLU(alpha=0.3),strides=1):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(filter, ksize, padding=padding, activation=activation, strides=strides)
    
    def call(self, x):
        x = tf.pad(x, [[0, 0], [2, 2],[2, 2], [0, 0]], mode='SYMMETRIC')
        x = self.conv2d(x)
        return x