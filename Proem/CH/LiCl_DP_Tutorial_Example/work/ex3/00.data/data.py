import dpdata 
import numpy as np

data = dpdata.LabeledSystem('../../ex2/01.md', fmt = 'abacus/md') 

# random choose 100 index for validation_data
index_validation = np.random.choice(len(data),size=100,replace=False)     
# other indexes are training_data
index_training = list(set(range(len(data)))-set(index_validation))       
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

# all training data put into directory:"training_data" 
data_training.to_deepmd_npy('training_data')
data_training.to_deepmd_raw('training_data')               
# all validation data put into directory:"validation_data"
data_validation.to_deepmd_npy('validation_data') 
data_validation.to_deepmd_raw('validation_data')  

print('# the data contains %d frames' % len(data))     
print('# the training data contains %d frames' % len(data_training)) 
print('# the validation data contains %d frames' % len(data_validation)) 