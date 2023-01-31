from models import overfitNN
from library import readYUV420, RGB2YUV, YUV2RGB, writeYUV420
import numpy as np 
import random 
import tensorflow as tf 

videoFile = 'data/test.yuv'
resolution = (512,512)
epochs = 50

Y,U,V = readYUV420(videoFile, resolution, True)
YUV = np.stack([Y,U,V], -1)
RGB = YUV2RGB(YUV)

NNModel = overfitNN().model()
LossFunc = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

m = tf.keras.metrics.MeanSquaredError()

for _epoch in range(epochs):
    _frameRange = np.arange(0, RGB.shape[0], 1)
    random.shuffle(_frameRange)
    for _step in _frameRange:
        xIn = np.expand_dims(np.array([_step]),0)
        with tf.GradientTape() as tape:
            xOut = NNModel(xIn, training=True)
            _loss = LossFunc(xOut, RGB[[_step]])
        gradients = tape.gradient(_loss, NNModel.trainable_weights)
        opt.apply_gradients(zip(gradients, NNModel.trainable_weights))
        m.update_state(xIn, xOut)
        print(_step)
    print(f"Epoch: {_epoch}, loss: {m.result().numpy()}")
    m.reset_state()
    # Evaluate Performance on Normal 
    RGBOut = np.zeros_like(RGB)
    for _out in range(0, RGB.shape[0], 1):
        RGBOut[_out] = NNModel(_out/RGB.shape[0], training=False)[0]
    RGBOut = np.rint(RGBOut * 255)
    YUVOut = RGB2YUV(RGBOut)
    YUVOut = np.rint(np.clip(YUVOut,0, 255)).astype(np.uint8)
    writeYUV420(f'results/{_epoch}.yuv', YUVOut[:,:,:,0],YUVOut[:,:,:,1],YUVOut[:,:,:,2])
    # Evaluate Performance on 2x SlowDown
    RGBOut = np.zeros((int(2*RGB.shape[0]), RGB.shape[1], RGB.shape[2], RGB.shape[3]))
    for _out in range(0, int(2*RGB.shape[0]), 1):
        RGBOut[_out] = NNModel(_out/(2*RGB.shape[0]), training=False)[0]
    RGBOut = np.rint(RGBOut * 255)
    YUVOut = RGB2YUV(RGBOut)
    YUVOut = np.rint(np.clip(YUVOut,0, 255)).astype(np.uint8)
    writeYUV420(f'resultsSlowDown/{_epoch}_2xSlower.yuv', YUVOut[:,:,:,0],YUVOut[:,:,:,1],YUVOut[:,:,:,2])

