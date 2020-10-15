import numpy as np
import pandas as pd
from Normalize import normalize
from Neural_Network import train_nn
from Evaluation import evaln

def find_string(l1,s):
    matched_indexes = []
    i = 0
    length = len(l1)

    while i < length:
        if s == l1[i]:
            matched_indexes.append(i)
        i += 1
    return np.asarray(matched_indexes)


an = 0
if an == 1:
    wb = pd.ExcelFile('.\KDD_cup99.xlsx')  # Read Excel file
    df1 = wb.parse('Sheet1')
    data = df1.values  # Get the values oif sheet1
    for n1 in range(data.shape[1]):
        if isinstance(data[0,n1], str):
            if n1 == data.shape[1] - 1:
                val = np.ones((data.shape[0]))
                ind = find_string(data[:, n1], 'normal')
                if ind.any(): val[ind] = int(0)
                data[:, n1] = val
            else:
                u = np.unique(data[:, n1])
                for n2 in range(len(u)):
                    ind = find_string(data[:, n1], u[n2])
                    data[ind, n1] = n2
    np.save('data.npy', data)
else:
    data = np.load('data.npy')

feat = data[:,0:data.shape[1] - 2]
tar = data[:, data.shape[1] - 1]
feat = normalize(feat)

per = round(feat.shape[0] * 0.70)  # 70% of learning
train_data = feat[0:per - 1, :]
train_target = tar[0:per - 1]
test_data = feat[per:per + data.shape[0]-1, :]
test_target = tar[per:per + data.shape[0]-1]
act = test_target

pred, net = train_nn(train_data, train_target, test_data, 10)
act.astype(bool)
pred.astype(bool)
EVAL = evaln([pred], [act])

print('Accuracy  Sensitivity  Specificity  Precision  FPR  FNR  NPV  FDR   F1-Score  MCC')
print(EVAL[4:])