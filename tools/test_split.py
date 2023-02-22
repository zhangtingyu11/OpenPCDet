import random

TOTAL_NUM = 7481
TRAINING_RATIO = 0.8

if __name__ == '__main__':
    training_num = int(TOTAL_NUM*TRAINING_RATIO)
    val_num = TOTAL_NUM - training_num
    training_set = random.sample(list(range(TOTAL_NUM)), training_num)
    training_set.sort()
    training_set = set(training_set)
    f = open('train.txt', 'w')
    for i, idx in enumerate(training_set):
        f.write(str(idx).zfill(6))
        if(i != len(training_set)-1):
            f.write('\n')
    f.close()
    val_set = set()
    for i in range(TOTAL_NUM):
        if(i not in training_set):
            val_set.add(i)
    f = open('val.txt', 'w')
    for i, idx in enumerate(val_set):
        f.write(str(idx).zfill(6))
        if(i != len(val_set)-1):
            f.write('\n')
    f.close()
        
            
        