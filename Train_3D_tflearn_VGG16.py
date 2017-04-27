import tflearn
import numpy as np
import cnn_3d_network as cnn3d
import os
import tensorflow as tf
import datetime

BATCH_SIZE = 8
N_EPOCH = 1
much_data = np.load('./preprocessing_model/muchdata_lungs_structures_128*128*128_bk.npy')

# Normalization
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

t1 = datetime.datetime.now()
X = [np.expand_dims(line[0],-1) for line in much_data]
Y = [line[1] for line in much_data]

# Building 'VGG16'
#with tf.device('/gpu:1'),tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
network = cnn3d.create_cnn_3d_VGG16()
model_file = "./model_tflean/cnn_3d_vgg16.ckpt"
# Training
model = tflearn.DNN(network, checkpoint_path=model_file,
                    max_checkpoints=2, tensorboard_verbose=2,tensorboard_dir="./logs")

if os.path.isfile(model_file + ".index"):
    model.load(model_file)
    print('load modle:' + model_file)

model.fit(X, Y, n_epoch=N_EPOCH, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=BATCH_SIZE, snapshot_step=10,
          snapshot_epoch=True)

model.save(model_file)
t2 = datetime.datetime.now()
print('The used time is:' + str(t2-t1))