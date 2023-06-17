import numpy as np
import random

train_data_no   = 800
val_data_no     = 100
test_data_no    = 100

data_seed = 1

train_data_x  = []
val_data_x    = []
test_data_x   = []

# Fill x coordinates with random values between 0 (inclued) and 1 (excluded)
random.seed(data_seed)
for x in range(train_data_no):
    train_data_x.append(random.random())

for x in range(val_data_no):
    val_data_x.append(random.random())

for x in range(test_data_no):
    test_data_x.append(random.random())

# Create output files
with open('training_data.txt', 'w') as f:
    for x_data in train_data_x:
        f.write(str(x_data) + ' ' + str(np.sin(x_data*2*np.pi)) + '\n')

with open('validation_data.txt', 'w') as f:
    for x_data in val_data_x:
        f.write(str(x_data) + ' ' + str(np.sin(x_data*2*np.pi)) + '\n')

with open('test_data.txt', 'w') as f:
    for x_data in test_data_x:
        f.write(str(x_data) + ' ' + str(np.sin(x_data*2*np.pi)) + '\n')