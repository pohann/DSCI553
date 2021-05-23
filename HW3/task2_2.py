# HW3 Task2-2
# Po-Han Chen USCID: 3988044558
# Mar 03 2021


from pyspark import SparkContext, SparkConf
import os
import sys
import time
import json

# configuration on local machine
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
# parse command line arguments
folder_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]


# read rows
def read_train(x):
    row_list = x.split(",")
    return row_list[1], row_list[0], row_list[2]


def read_test(x):
    row_list = x.split(",")
    return row_list[1], row_list[0]


"""
Start timer
"""
start_time = time.time()
conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
# read data
# train_RDD: (business_id, user_id, stars)
# test_RDD: (business_id, user_id)
train_RDD = sc.textFile(folder_path+"yelp_train.csv").filter(lambda x: not x.startswith("user_id")).map(
    read_train)
test_RDD = sc.textFile(test_path).filter(lambda x: not x.startswith("user_id")).map(
    read_test)

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

"""
Getting features
"""
# read user.json
user_data = []
with open(folder_path + "user.json") as in_file:
    while True:
        data = in_file.readlines(15000000)
        if not data:
            break
        user_RDD = sc.parallelize(data).map(lambda x: json.loads(x)).map(
            lambda x: (x["user_id"], x["average_stars"], x["review_count"])).filter(lambda x: x[0] in uid_dict).map(
            lambda x: (uid_dict[x[0]], float(x[1]), float(x[2])))
        user_data += user_RDD.collect()
# print(user_frame.head())
# read business.json
business_RDD = sc.textFile(folder_path+"business.json").map(lambda x: json.loads(x)).map(
    lambda x: (x["business_id"], x["stars"], x["review_count"])).filter(lambda x: x[0] in bid_dict).map(
    lambda x: (bid_dict[x[0]], float(x[1]), float(x[2])))
business_data = business_RDD.collect()
print(business_data[0])
# preprocessing data with pandas
import pandas as pd
import numpy as np
user_frame = pd.DataFrame(user_data, columns=["user_id", "average_stars", "review_count"])
business_frame = pd.DataFrame(business_data, columns=["business_id", "stars", "review_count"])
user_frame.set_index("user_id", inplace=True)
business_frame.set_index("business_id", inplace=True)
print(user_frame.head())
bid_index = set(business_frame.index)
uid_index = set(user_frame.index)
"""
Model training
"""


# get training data X and Y
def get_matrix_train(x, user_frame, business_frame):
    if x[1][0] not in uid_index or x[0] not in bid_index:
        return [np.nan]*(len(business_frame.columns)+len(user_frame.columns))
    else:
        return list(user_frame.loc[x[1][0]]) + list(business_frame.loc[x[0]])


def get_matrix_test(x, user_frame, business_frame):
    if x[1] not in uid_index or x[0] not in bid_index:
        return [np.nan]*(len(business_frame.columns)+len(user_frame.columns))
    else:
        return list(user_frame.loc[x[1]]) + list(business_frame.loc[x[0]])


def get_y(x):
    return x[1][1]


X = np.array(train_RDD.map(lambda x: get_matrix_train(x, user_frame, business_frame)).collect())
print(X[0:5])
Y = np.array(train_RDD.map(get_y).collect())
print(Y[0])

# train the model
import xgboost as xgb
xgbr = xgb.XGBRegressor(verbosity=0, n_estimators=30, random_state=1, max_depth=7)
xgbr.fit(X, Y)

# make predictions
X_test = np.array(test_RDD.map(lambda x: get_matrix_test(x, user_frame, business_frame)).collect())
Y_pred = xgbr.predict(X_test)
print(type(Y_pred))
# write results
res = "user_id, business_id, prediction\n"
rows = test_RDD.collect()
for i in range(len(rows)):
    res += uid_dict_reversed[rows[i][1]] + "," + bid_dict_reversed[rows[i][0]] + "," + str(Y_pred[i]) + "\n"

with open(output_path, "w") as out_file:
    out_file.writelines(res)

"""
End timer
"""
duration = time.time() - start_time
print("Duration: "+str(duration))

"""
See results
"""
import numpy as np
# calculate RMSE
with open("task2_2.csv") as in_file:
    guess = in_file.readlines()[1:]
with open("data/yelp_val.csv") as in_file:
    ans = in_file.readlines()[1:]
res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
RMSE = 0
for i in range(len(guess)):
    diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
    RMSE += diff**2
    if abs(diff) < 1:
        res["<1"] = res["<1"] + 1
    elif 2 > abs(diff) >= 1:
        res["1~2"] = res["1~2"] + 1
    elif 3 > abs(diff) >= 2:
        res["2~3"] = res["2~3"] + 1
        # if diff > 0:
        #     large_small["large"] = large_small["large"] + 1
        # else:
        #     large_small["small"] = large_small["small"] + 1
        # print(guess[i].split(","))
        # print(ans[i].split(","))
        # print("========")
    elif 4 > abs(diff) >= 3:
        res["3~4"] = res["3~4"] + 1
    else:
        res["4~5"] = res["4~5"] + 1
RMSE = (RMSE/len(guess))**(1/2)
print("RMSE: "+str(RMSE))
prediction = np.array([float(gg.split(',')[2]) for gg in guess])
print("Prediction mean: " + str(prediction.mean()))
print("Prediction std:" + str(prediction.std()))
ground = np.array([float(gg.split(',')[2]) for gg in ans])
print("Answer mean: "+str(ground.mean()))
print("Answer std: "+str(ground.std()))
