from sklearn import preprocessing
import numpy as np
import pandas as pd
from recommenders.models.sar import SAR

def np_to_pandas_nodelete_itemfirst(user_id, x):
    np_list = [[0, 0, 0, 0]]
    timestep = 90000000
    # x = binarize(x, 0)

    user_id_end = user_id + x.shape[0]
    k = 0
    for i in range(user_id, user_id_end):
        for j in range(x.shape[1]):
            for j in range(x.shape[1]):
                if x[k][j] != 0:
                    np_list.append([int(i), j, x[k][j], 19981015])
                    timestep += 1
        k += 1
    np_list = np.array(np_list)
    return np_list[1:], user_id_end


label_data = np.loadtxt("swin_train_gt.csv", delimiter=',')
print(len(label_data))
label_data_numpy, _ = np_to_pandas_nodelete_itemfirst(0, label_data)
label_data_pandas = pd.DataFrame(label_data_numpy, columns=["userID", "itemID", "rating", "timestamp"])
label_data_pandas['rating'] = label_data_pandas['rating'].astype(np.float32)
train = label_data_pandas

model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",
    time_decay_coefficient=30,
    timedecay_formula=True,
    normalize=False
)

model.fit(train)
top_k = model.recommend_k_items(train, top_k=51, remove_seen=False)

all_predictions = top_k.sort_values(['userID', 'itemID'], ascending=True).groupby('userID').head(51)

np_predictions = all_predictions['prediction'].to_numpy()
np_predictions = np_predictions.reshape(33216, 51)

lowest = np.amin(np_predictions)
np_predictions += lowest

np_predictions = preprocessing.minmax_scale(np_predictions)
np.savetxt('swin_train_sar_scales.csv', np_predictions, delimiter=',')