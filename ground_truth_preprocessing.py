import pandas as pd
import numpy as np
import csv


full_data = pd.read_csv('./data/full_rle_1234.csv').fillna('')
total_nameId = full_data['ImageId']

with open('./data/unique_rle_1234.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageId', 'rle_1', 'rle_2', 'rle_3', 'rle_4'])

    unique_nameId = []
    for i in range(len(total_nameId)):
        nameId = total_nameId[i]
        if nameId not in unique_nameId:
            a = np.array(total_nameId)
            idx = np.where(a == nameId)
            idx = idx[0]
            unique_nameId.append(nameId)
            
            rle_1, rle_2, rle_3, rle_4 = '', '', '', ''
            for n in range(len(idx)):
                data = full_data.iloc[idx[n]]
                if data[1] != '': rle_1 = data[1]
                if data[2] != '': rle_2 = data[2]
                if data[3] != '': rle_3 = data[3]
                if data[4] != '': rle_4 = data[4]

            writer.writerow([nameId, rle_1, rle_2, rle_3, rle_4])