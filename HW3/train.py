# HW3 train
# Po-Han Chen USCID: 3988044558
# Mar 03 2021
# This script aims at finding out when to use which method with a classification problem

from pyspark import SparkContext, SparkConf
import json
import numpy as np

# configuration on local machine
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def read_test(x):
    row_list = x.split(",")
    return row_list[1], row_list[0]

def read_train(x):
    row_list = x.split(",")
    return row_list[1], row_list[0], row_list[2]


"""
Compare two output files to know who is doing better and when
"""
conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# test_RDD = sc.textFile("../resource/asnlib/publicdata/yelp_val_in.csv").filter(lambda x: not x.startswith("user_id")).map(read_test)
test_RDD = sc.textFile("data/yelp_val_in_small.csv").filter(lambda x: not x.startswith("user_id")).map(read_test)
# train_RDD = sc.textFile("../resource/asnlib/publicdata/yelp_train.csv").filter(lambda x: not x.startswith("user_id")).map(read_train)
train_RDD = sc.textFile("data/yelp_train.csv").filter(lambda x: not x.startswith("user_id")).map(read_train)

# get dictionary of business_id
bid_dict_train = set(train_RDD.map(lambda x: x[0]).distinct().collect())
bid_dict_test = set(test_RDD.map(lambda x: x[0]).distinct().collect())
bid_dict = bid_dict_train.union(bid_dict_test)
bid_dict = dict(zip(bid_dict, range(len(bid_dict))))
bid_dict_reversed = dict(zip(range(len(bid_dict)), bid_dict))
# get dictionary of user_id
uid_dict_train = set(train_RDD.map(lambda x: x[1]).distinct().collect())
uid_dict_test = set(test_RDD.map(lambda x: x[1]).distinct().collect())
uid_dict = uid_dict_train.union(uid_dict_test)
uid_dict = dict(zip(uid_dict, range(len(uid_dict))))
uid_dict_reversed = dict(zip(range(len(uid_dict)), uid_dict))
# use dictionaries to save space
train_RDD = train_RDD.map(lambda x: (bid_dict[x[0]], (uid_dict[x[1]], int(x[2][0])))).cache()
test_RDD = test_RDD.map(lambda x: (bid_dict[x[0]], uid_dict[x[1]])).cache()

def d_mean(x):
    """
    deduct mean for each item
    :param x:
    :return:
    """
    mean = sum([xx[1] for xx in x[1]]) / len(x[1])
    res = [(xx[0], xx[1] - mean) for xx in x[1]]
    return x[0], dict([(xx[0], xx[1]) for xx in x[1]]), dict(res)


# train_RDD_bid: bid, {uid: stars-mean, ...}. for a bid, see who have rated this item and what's the rating
# train_RDD_bid_origin: bid, {uid: stars, ...}. for a bid, see who have rated this item and what's the rating
train_RDD_bid_ = train_RDD.groupByKey().map(lambda x: (x[0], list(x[1]))).map(d_mean).collect()
train_RDD_bid = dict(zip([xx[0] for xx in train_RDD_bid_], [xx[2] for xx in train_RDD_bid_]))
train_RDD_bid_origin = dict(zip([xx[0] for xx in train_RDD_bid_], [xx[1] for xx in train_RDD_bid_]))
# train_RDD_uid: uid, [(bid, stars), ...] for a uid, see what items s/he have rated
train_RDD_uid = train_RDD.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
train_RDD_uid = dict(zip([xx[0] for xx in train_RDD_uid], [xx[1] for xx in train_RDD_uid]))

"""
Getting features
"""
# read user.json
user_data = []
# with open("../resource/asnlib/publicdata/user.json") as in_file:
with open("data/user.json") as in_file:
    while True:
        data = in_file.readlines(15000000)
        if not data:
            break
        user_RDD = sc.parallelize(data).map(lambda x: json.loads(x)).map(
            lambda x: (x["user_id"], x["average_stars"], x["review_count"])).filter(lambda x: x[0] in uid_dict).map(
            lambda x: (uid_dict[x[0]], float(x[1]), float(x[2])))
        user_data += user_RDD.collect()
# read business.json
business_RDD = sc.textFile("data/business.json").map(lambda x: json.loads(x)).map(
    lambda x: (x["business_id"], x["stars"], x["review_count"])).filter(lambda x: x[0] in bid_dict).map(
    lambda x: (bid_dict[x[0]], float(x[1]), float(x[2])))
# business_RDD = sc.textFile("../resource/asnlib/publicdata/business.json").map(lambda x: json.loads(x)).map(
#     lambda x: (x["business_id"], x["stars"], x["review_count"])).filter(lambda x: x[0] in bid_dict).map(
#     lambda x: (bid_dict[x[0]], float(x[1]), float(x[2])))
business_data = business_RDD.collect()

# standard deviation of user's rating
uid_std = []
for kk in train_RDD_uid:
    stars = np.array([xx[1] for xx in train_RDD_uid[kk]])
    uid_std.append([kk, stars.std()])
print(uid_std[1])
# standard deviation of business's rating
bid_std = []
for kk in train_RDD_bid_origin:
    stars = np.array([xx for xx in train_RDD_bid_origin[kk].values()])
    bid_std.append([kk, stars.std()])
print(bid_std[1])
"""
Model training
"""
# preprocessing data with pandas
import pandas as pd
import numpy as np
user_frame = pd.DataFrame(user_data, columns=["user_id", "average_stars", "review_count"])
business_frame = pd.DataFrame(business_data, columns=["business_id", "stars", "review_count"])
uid_std_df = pd.DataFrame(uid_std, columns=["user_id", "std"])
bid_std_df = pd.DataFrame(bid_std, columns=["business_id", "std"])
user_frame = user_frame.set_index("user_id").join(uid_std_df.set_index("user_id"))
business_frame = business_frame.set_index("business_id").join(bid_std_df.set_index("business_id"))
print(business_frame.head())
print(user_frame.head())

# get training data X and Y
def get_matrix_test(x, user_frame, business_frame):
    return list(user_frame.loc[x[1]]) + list(business_frame.loc[x[0]])


# get X
X_test = np.array(test_RDD.map(lambda x: get_matrix_test(x, user_frame, business_frame)).collect())
# get Y
with open("task2_1_small.csv") as in_file:
    res2_1 = in_file.readlines()[1:]
with open("task2_2_small.csv") as in_file:
    res2_2 = in_file.readlines()[1:]
with open("data/yelp_val_small.csv") as in_file:
# with open("../resource/asnlib/publicdata/yelp_val.csv") as in_file:
    ans = in_file.readlines()[1:]
Y = []
for i in range(len(ans)):
    if abs(float(res2_2[i].split(",")[2]) - float(ans[i].split(",")[2])) < abs(float(res2_1[i].split(",")[2]) - float(ans[i].split(",")[2])):
        Y.append(1)
    else:
        Y.append(0)
Y = np.array(Y)
print(Y)
# res_Y = ""
# for yy in Y:
#     res_Y += str(yy) + " "
# with open("Y.csv", "w") as out_file:
#     out_file.write(res_Y)
# train the model
import xgboost as xgb
from joblib import dump, load
xgbc = xgb.XGBClassifier(verbosity=0, n_estimators=50,  use_label_encoder=False)
xgbc.fit(X_test, Y)
dump(xgbc, "model.joblib")
print(xgbc.score(X_test, Y))
