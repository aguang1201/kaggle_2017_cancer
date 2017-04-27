import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

IMG_SIZE_PX = 128
keep_rate = 0.5
num_class = 2

def create_cnn_3d_network():
    # Building 'AlexNet'
    network = input_data(shape=[None, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])
    network = conv_3d(network, 32, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')
    network = max_pool_3d(network, 3, strides=2)

    network = conv_3d(network, 64, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')
    network = max_pool_3d(network, 3, strides=2)

    network = conv_3d(network, 128, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 256, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')
    network = max_pool_3d(network, 3, strides=2)

    network = fully_connected(network, 2048)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')
    network = dropout(network, keep_rate)

    network = fully_connected(network, 2048)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')
    network = dropout(network, keep_rate)

    network = fully_connected(network, num_class)

    network = regression(network, optimizer='adam',
                         loss='softmax_categorical_crossentropy',
                         learning_rate=0.0001)

    return network

def create_cnn_3d_alex():

    #img_prep = ImagePreprocessing()
    #img_prep.add_featurewise_zero_center(mean=0.25)

    network = input_data(shape=[None, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    network = conv_3d(network, 96, 11, strides=4, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 3, strides=2)

    network = conv_3d(network, 256, 5, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 3, strides=2)

    network = conv_3d(network, 384, 3, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 384, 3, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 256, 3, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 3, strides=2)

    network = fully_connected(network, 4096, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = dropout(network, keep_rate)

    network = fully_connected(network, 4096, regularizer='L2')
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = dropout(network, keep_rate)

    output = fully_connected(network, num_class, activation='softmax')

    network = regression(output, optimizer='adam',
                             loss='categorical_crossentropy',
                         learning_rate=0.0001)
    return network

def create_cnn_3d_VGG16():

    #img_prep = ImagePreprocessing()
    #img_prep.add_featurewise_zero_center(mean=0.25)

    network = input_data(shape=[None, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    network = conv_3d(network, 64, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 64, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 2, strides=2)

    network = conv_3d(network, 64, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 32, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 2, strides=2)

    network = conv_3d(network, 256, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 256, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 256, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 2, strides=2)

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 2, strides=2)

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = conv_3d(network, 512, 3)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = max_pool_3d(network, 2, strides=2)

    network = fully_connected(network, 2048)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = dropout(network, keep_rate)

    network = fully_connected(network, 2048)
    network = tflearn.activation(tflearn.batch_normalization(network), activation='relu')

    network = dropout(network, keep_rate)

    output = fully_connected(network, num_class, activation='softmax')

    network = regression(output, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network
