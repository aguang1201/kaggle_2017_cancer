import pandas as pd
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--in_num', type=str, default='8',help='input file number')
parser.add_argument('--out_num', type=str, default='2',help='output file number')

args = parser.parse_args()
input_file_num = args.in_num
output_file_num = args.out_num

results_list = ['id'+','+'cancer'+'\n']
results_mean_1 = pd.read_csv('./output/results_mean_1.csv')
results_input = pd.read_csv('./output/results_{}.csv'.format(input_file_num))

prediction_mean_1 = results_mean_1['cancer']
prediction_input = results_input['cancer']
patients_input = results_input['id']

prediction_mean_1 = map(lambda x:sum(x)/2,zip(prediction_mean_1,prediction_input))

for patient,predict in zip(patients_input,prediction_mean_1):
    results_list.append(patient+','+str(predict)+'\n')

with open('./output/results_mean_{}.csv'.format(output_file_num),'a') as results_file:
    results_file.writelines(results_list)