import tflearn
import numpy as np
import cnn_3d_network as cnn3d
import os
import datetime
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

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
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.add_to_collection('graph_config', config)
def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            network = cnn3d.create_cnn_3d_alex()
            #model = tflearn.DNN(network, checkpoint_path=model_file, best_checkpoint_path=best_model_file,
            #                    max_checkpoints=2, tensorboard_verbose=2, tensorboard_dir="./logs")

            global_step = tf.Variable(0)
            #train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        with sv.managed_session(server.target) as sess:
                # Force all Variables to reside on the CPU.
            #with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
            model = tflearn.DNN(network)
            model.fit(X, Y, n_epoch=N_EPOCH, validation_set=0.1, shuffle=True,
                      show_metric=True, batch_size=BATCH_SIZE, snapshot_step=20, snapshot_epoch=True)

if __name__ == "__main__":
    tf.app.run()
t2 = datetime.datetime.now()
print('The used time is:' + str(t2))
print('The used time is:' + str(t2-t1))