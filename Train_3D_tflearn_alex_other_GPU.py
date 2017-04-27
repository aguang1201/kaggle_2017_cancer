import tflearn
import numpy as np
import cnn_3d_network as cnn3d
import os
import datetime
import tensorflow as tf

BATCH_SIZE = 64
N_EPOCH = 1
IMG_SIZE_PX = 128
much_data = np.load('./preprocessing_model/muchdata_lungs_fill_128*128*128.npy')
t1 = datetime.datetime.now()
X = [[line[0]] for line in much_data]
X = np.array(X).reshape([-1,128,128,128,1])
Y = [line[1] for line in much_data]
model_file = "./model_tflean/cnn_3d_alex_lungs_fill.ckpt"
best_model_file = "./model_tflean_best/cnn_3d_alex_lungs_fill.ckpt"
# Training
#with tf.Graph().as_default():
#tflearn.init_graph(gpu_memory_fraction=1,soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.add_to_collection('graph_config', config)
#session = tf.Session(config=config)

network = cnn3d.create_cnn_3d_alex()
model = tflearn.DNN(network, checkpoint_path=model_file, best_checkpoint_path=best_model_file,
                    max_checkpoints=2, tensorboard_verbose=2, tensorboard_dir="./logs")
if os.path.isfile(model_file + ".index"):
    model.load(model_file)
    print('load the best modle:' + model_file)

with tf.device('/gpu:0'):
    # Force all Variables to reside on the CPU.
    with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
        model.fit(X, Y, n_epoch=N_EPOCH, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=BATCH_SIZE, snapshot_step=20, snapshot_epoch=True)
tf.get_variable_scope().reuse_variables()
with tf.device('/gpu:1'):
    # Force all Variables to reside on the CPU.
    with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
        model.fit(X, Y, n_epoch=N_EPOCH, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=BATCH_SIZE, snapshot_step=20, snapshot_epoch=True)
tf.get_variable_scope().reuse_variables()

model.save(model_file)
t2 = datetime.datetime.now()
print('The used time is:' + str(t2))
print('The used time is:' + str(t2-t1))