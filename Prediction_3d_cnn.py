import tflearn
import cnn_3d_network as cnn3d
import os
import numpy as np
import datetime
import math
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--num',type=int,default=0,help='Number of steps to run prediction.')
parser.add_argument('--out',type=str,default='14',help='output file number')
args = parser.parse_args()
num = int(args.num)
out_num = args.out

def prediction(images):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.add_to_collection('graph_config', config)
    with tf.device('/gpu:0'):
        network = cnn3d.create_cnn_3d_alex()
    #with tf.device('/cpu:0'):
    model = tflearn.DNN(network)
    model_file = "./model_tflean_best/cnn_3d_alex_lungs_fill.ckpt-19619"
    #if os.path.isfile(model_file + ".index"):
    model.load(model_file)
    print('load modle:' + model_file)
    return model.predict(images)

t1 = datetime.datetime.now()

batch_size = 90
#count=math.ceil(len(X)/90)

if num==0:
    results_list = ['id'+','+'cancer'+'\n']
else:
    results_list = []
test_data = np.load('./preprocessing_model/muchdata_lungs_fill_test_128*128*128.npy')
X = np.transpose(test_data, (0, 2, 3, 4, 1))
results = prediction(X[num*batch_size:(num+1)*batch_size])

INPUT_FOLDER = '/home/wisdom/deeplearningdata/kaggle_2017_cancer/stage2/'
#INPUT_FOLDER = '/home/wisdom/deeplearningdata/kaggle_2017_cancer/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
for patient,predict in zip(patients[num*batch_size:(num+1)*batch_size],results):
    results_list.append(patient+','+str(predict[1])+'\n')

with open('./output/results_{}.csv'.format(out_num),'a') as results_file:
    results_file.writelines(results_list)

t2 = datetime.datetime.now()
print('The used time is:' + str(t2))
print('The used time is:' + str(t2-t1))