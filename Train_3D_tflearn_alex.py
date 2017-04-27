import tflearn
import numpy as np
import cnn_3d_network as cnn3d
import os
import datetime
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of Batch.')
flags.DEFINE_integer('N_EPOCH', 400, 'Number of steps to run trainer.')
flags.DEFINE_integer('IMG_SIZE_PX', 128, 'Size of Image.')
n_epoch=FLAGS.N_EPOCH
batch_size=FLAGS.BATCH_SIZE

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.add_to_collection('graph_config', config)

#much_data = np.load('./preprocessing_model/muchdata_lungs_fill_add_cancer_128*128*128.npy')
much_data = np.load('./preprocessing_model/muchdata_lungs_fill_128*128*128.npy')

t1 = datetime.datetime.now()

X = [np.expand_dims(line[0], -1) for line in much_data]
#X = [[line[0]] for line in much_data]
#X = np.array(X).reshape([-1,128,128,128,1])
Y = [line[1] for line in much_data]
model_file = "./model_tflean/cnn_3d_alex_lungs_fill.ckpt"
#best_model_file = "./model_tflean_best/cnn_3d_alex_lungs_fill_best_val"
# Building 'AlexNet'
#with tf.device('/gpu:1'):
network = cnn3d.create_cnn_3d_alex()

# Training
model = tflearn.DNN(network, checkpoint_path=model_file,max_checkpoints=2,
                        tensorboard_verbose=2,tensorboard_dir="./logs")

if os.path.isfile(model_file + ".index"):
    model.load(model_file)
    print('load modle:' + model_file)

model.fit(X, Y, n_epoch=n_epoch, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=batch_size, snapshot_epoch=True)

model.save(model_file)
t2 = datetime.datetime.now()
print('The used time is:' + str(t2))
print('The used time is:' + str(t2-t1))