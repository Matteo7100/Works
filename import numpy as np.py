import numpy as np
from collections import Counter



y_pred = [[0,5,1,4],[0,5,1,4],[1,4,8,6]]
y_pred_string = [''.join(map(str, seq)) for seq in y_pred]
print(y_pred_string)
a = Counter(y_pred_string)
print(a)
esponente = 0
x_values = np.array(list(a.values()))
print(x_values)
