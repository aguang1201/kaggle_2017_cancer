import numpy as np
'''
much_data_1 = np.load('./preprocessing_model/muchdata_1.npy')
much_data_2 = np.load('./preprocessing_model/muchdata_2.npy')
much_data_3 = np.load('./preprocessing_model/muchdata_3.npy')
much_data_4 = np.load('./preprocessing_model/muchdata_4.npy')
much_data_5 = np.load('./preprocessing_model/muchdata_5.npy')
much_data_6 = np.load('./preprocessing_model/muchdata_6.npy')
much_data_7 = np.load('./preprocessing_model/muchdata_7.npy')
much_data_8 = np.load('./preprocessing_model/muchdata_8.npy')
much_data_9 = np.load('./preprocessing_model/muchdata_9.npy')
much_data_10 = np.load('./preprocessing_model/muchdata_10.npy')
'''
much_data_1 = np.load('./preprocessing_model/muchdata_lungs_structures_128*128*128.npy')
much_data_2 = np.load('./preprocessing_model/muchdata_lungs_structures_solution_128*128*128.npy')
#much_data = much_data_1 + much_data_2 + much_data_3 + much_data_4 + much_data_5 + much_data_6 + much_data_7 + much_data_8 + much_data_9 + much_data_10
much_data = np.concatenate((much_data_1,much_data_2),axis=0)
np.save('./preprocessing_model/muchdata_lungs_structures_train_validation_128*128*128.npy', much_data)
print(str(len(much_data)))