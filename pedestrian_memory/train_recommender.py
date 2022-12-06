import sys
import os
import surprise
import papermill as pm
import scrapbook as sb
import pandas as pd
import numpy as np
import cornac
from recommenders.utils.constants import SEED
from recommenders.models.cornac.cornac_utils import predict_ranking

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k,
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions
from sklearn import preprocessing
print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))

import logging
import numpy as np
import pandas as pd
import scrapbook as sb
from sklearn.preprocessing import minmax_scale

from recommenders.utils.timer import Timer

from recommenders.models.sar import SAR
import sys

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

def np_to_pandas(user_id, x):
    np_list = [[0, 0, 0]]
    timestep = 90000000
    user_id_end = user_id + x.shape[0]
    k = 0
    for i in range(user_id, user_id_end):
        for j in range(x.shape[1]):
            np_list.append([int(i), j, x[k][j]*5])
            timestep += 1
        k += 1
    np_list = np.array(np_list)
    return np_list[1:], user_id_end

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

label_data = np.loadtxt("SWIN_TRAIN_PAR_GT.csv", delimiter=',')

label_data_numpy, _ = np_to_pandas_nodelete_itemfirst(0, label_data)
label_data_pandas = pd.DataFrame(label_data_numpy, columns=["userID", "itemID", "rating", "timestamp"])
label_data_pandas['rating'] = label_data_pandas['rating'].astype(np.float32)
train = label_data_pandas

train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

bpr = cornac.models.BPR(
    k=200,
    max_iter=100,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)
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

with Timer() as t:
    model.fit(train)
print("Took {} seconds for training.".format(t))
with Timer() as test_time:
    top_k = model.recommend_k_items(train, top_k=51, remove_seen=False)

print("Took {} seconds for prediction.".format(test_time.interval))
print("Took {} seconds for prediction.".format(t))
all_predictions = top_k.sort_values(['userID', 'itemID'],ascending=True).groupby('userID').head(51)

print(all_predictions.head(50))

np_predictions = all_predictions['prediction'].to_numpy()
print(np_predictions.shape)
np_predictions = np_predictions.reshape(33268, 51)
np.savetxt('SWIN_TRAIN_SAR.csv', np_predictions, delimiter=',')

lowest = np.amin(np_predictions)
np_predictions += lowest

np_predictions = preprocessing.minmax_scale(np_predictions)
np.savetxt('SWIN_TRAIN_SAR_sclaed.csv', np_predictions, delimiter=',')
