import tensorflow as tf
import numpy as np
import tflearn
import os
import datetime

IMG_SIZE_PX = 128
n_classes = 2
batch_size = 10
x = tf.placeholder('float')
y = tf.placeholder('float')
keep_rate = 0.5
much_data = np.load('./preprocessing_model/muchdata.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-100]
validation_data = much_data[-100:]

# Normalization
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# Zero centering
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               #                                  64 features
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    conv1 = tf.nn.relu(tflearn.batch_normalization(conv3d(x, weights['W_conv1']) + biases['b_conv1']))
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(tflearn.batch_normalization(conv3d(conv1, weights['W_conv2']) + biases['b_conv2']))
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tflearn.batch_normalization(tf.matmul(fc, weights['W_fc']) + biases['b_fc']))
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 300
    saver = tf.train.Saver()
    model_path = "./model/cnn_3d.ckpt"
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        if os.path.isfile(model_path + '.index'):
            saver.restore(sess, model_path)
            print('load model' + model_path)

        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            print('epoch is %d' % (epoch))
            for num,data in enumerate(train_data):
                t3 = datetime.datetime.now()
                total_runs += 1
                if num%100==0:
                    print('epoch is %d,num is %d' % (epoch,num))
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    print(str(e))

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy_value = accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]})

            print('Accuracy:', accuracy_value)
            t4 = datetime.datetime.now()
            print('This epoch used time is:' + str(t4 - t3))
            if accuracy_value > 0.8:
                break

        print('Done. Finishing accuracy:')
        print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        print('fitment percent:', successful_runs / total_runs)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

# Run this locally:
t1=datetime.datetime.now()
train_neural_network(x)
t2=datetime.datetime.now()
print('Start time is:'+str(t1))
print('End time is:'+str(t2))
print('Used time is:'+str(t2-t1))