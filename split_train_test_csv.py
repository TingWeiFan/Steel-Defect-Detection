import pandas as pd
from sklearn.utils import shuffle


df = pd.read_csv('./data/aug_train.csv')
test_df = pd.read_csv('./data/test_rle_1234.csv')

all_data_id = df['ImageId']
test_data_id = test_df['ImageId']

image_id = []
rle_1, rle_2, rle_3, rle_4 = [], [], [], []
labels = [1, 2, 3, 4]
e = [rle_1, rle_2, rle_3, rle_4]
count1, count2, count3, count4 = 0, 0, 0, 0

for i in range(len(all_data_id)):
    if all_data_id[i] not in test_data_id:
        idx = labels.index(list(df.loc[i])[1])
        if idx == 0:
            image_id.append(list(df['ImageId'])[i])
            e[0].append(list(df.loc[i])[2])
            e[1].append('')
            e[2].append('')
            e[3].append('')

            count1 += 1
        if idx == 1:
            image_id.append(list(df['ImageId'])[i])
            e[1].append(list(df.loc[i])[2])
            e[0].append('')
            e[2].append('')
            e[3].append('')

            count2 += 1
        if idx == 2:
            image_id.append(list(df['ImageId'])[i])
            e[2].append(list(df.loc[i])[2])
            e[0].append('')
            e[1].append('')
            e[3].append('')

            count3 += 1
        if idx == 3:
            image_id.append(list(df['ImageId'])[i])
            e[3].append(list(df.loc[i])[2])
            e[0].append('')
            e[1].append('')
            e[2].append('')

            count4 += 1

datas = pd.DataFrame({'ImageId':image_id, 'rle_1':rle_1, 'rle_2':rle_2, 'rle_3':rle_3, 'rle_4':rle_4})
datas = datas.fillna('')
print(datas)

train_datas = shuffle(datas)
train_data = train_datas[:int(len(train_datas)*0.9)]
val_data = train_datas[int(len(train_datas)*0.9):]

train_data.to_csv('./data/train_rle_1234.csv', index=False)
val_data.to_csv('./data/val_rle_1234.csv', index=False)

#-----------------------------------------------------------------------------------------------#
file_lists = ['./data/train_rle_1234.csv', './data/val_rle_1234.csv', './data/test_rle_1234.csv']
for csv_file in file_lists:
    new_data = pd.read_csv(csv_file).fillna('')

    cls1, cls2, cls3, cls4 = 0, 0, 0, 0
    classes = [cls1, cls2, cls3, cls4]
    for i in range(len(new_data)):
        if list(new_data.loc[i])[1] != '': cls1 += 1

        if list(new_data.loc[i])[2] != '': cls2 += 1

        if list(new_data.loc[i])[3] != '': cls3 += 1

        if list(new_data.loc[i])[4] != '': cls4 += 1

    print('class 1:{}, class 2:{}, class 3:{}, class 4:{}'.format(cls1, cls2, cls3, cls4))