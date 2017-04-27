import pandas as pd
import argparse
import math

def compute(prediction):
    if prediction > 0.5:
        predict_result = prediction
    else:
        predict_result = 1 - prediction
    return math.log(predict_result)

#results_mean_1 = pd.read_csv('./output/results_mean_1.csv')
results_mean_1 = pd.read_csv('./output/results_5.csv')

prediction_mean_1 = results_mean_1['cancer']

predict_result = list(map(compute,prediction_mean_1))

#score = -math.log(0.9)
score = -1/len(predict_result)*sum(predict_result)

print('The scole is:' + str(score))
