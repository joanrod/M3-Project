import os

path_data = '../../MIT_small_train_1/MIT_small_train_1/'
path_train = path_data + 'test'
totalFiles = 0
totalDir = 0
for base, dirs, files in os.walk(path_train):
    print('Searching in : ',base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

print(totalFiles)