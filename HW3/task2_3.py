# HW3 Task2-3
# Po-Han Chen USCID: 3988044558
# Mar 03 2021


from pyspark import SparkContext, SparkConf
import os
import sys
import time
import json
import numpy as np

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
Model-based
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

# preprocessing data with pandas
import pandas as pd
user_frame = pd.DataFrame(user_data, columns=["user_id", "average_stars", "review_count"])
business_frame = pd.DataFrame(business_data, columns=["business_id", "stars", "review_count"])
uid_std_df = pd.DataFrame(uid_std, columns=["user_id", "std"])
bid_std_df = pd.DataFrame(bid_std, columns=["business_id", "std"])
user_frame_pred = user_frame.set_index("user_id")#.join(uid_std_df.set_index("user_id"))
business_frame_pred = business_frame.set_index("business_id")#.join(bid_std_df.set_index("business_id"))
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


X = np.array(train_RDD.map(lambda x: get_matrix_train(x, user_frame_pred, business_frame_pred)).collect())
print(X[0:5])
Y = np.array(train_RDD.map(get_y).collect())
print(Y[0])

# train the model
import xgboost as xgb
xgbr = xgb.XGBRegressor(verbosity=0, n_estimators=50, random_state=2, max_depth=7)
xgbr.fit(X, Y)

# make predictions
X_test = np.array(test_RDD.map(lambda x: get_matrix_test(x, user_frame_pred, business_frame_pred)).collect())
Y_pred = xgbr.predict(X_test)

print(type(Y_pred))
# write results
res_model = ""
rows = test_RDD.collect()
for i in range(len(rows)):
    res_model += uid_dict_reversed[rows[i][1]] + "," + bid_dict_reversed[rows[i][0]] + "," + str(Y_pred[i]) + "\n"


"""
Item-based
"""


def item_based(uid_bid, sim_dict):
    if uid_bid[1] not in train_RDD_uid:
        return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(3.75) + "\n"
    # find other items that the user have rated
    # rated: uid [(bid, stars), ...]
    rated = train_RDD_uid[uid_bid[1]]
    # when no other users have rated this items: use user's average rating
    if uid_bid[0] not in train_RDD_bid:
        ratings = [xx[1] for xx in rated]
        user_avg = sum(ratings)/len(ratings)
        return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(user_avg) + "\n"
    # to_rate: item to be rated
    # to_rate: {uid: stars-mean, ...}
    to_rate = train_RDD_bid[uid_bid[0]]
    to_rate_set = set(to_rate.keys())
    # calculate prediction
    sim_list = []
    rating_list = []
    # rated: uid [(bid, stars), ...]
    for rr in rated:
        neighbor = train_RDD_bid[rr[0]]
        intersect = set(neighbor.keys()).intersection(to_rate_set)
        if len(intersect) < 3:
            continue
        # first look up the dictionary of similarities
        pair = tuple(sorted([rr[0], uid_bid[0]]))
        if pair in sim_dict:
            sim = sim_dict[pair]
            # if sim < 0:
            #     continue
            sim_list.append(sim)
            rating_list.append(rr[1])
        else:
            sim_den = ((sum([to_rate[ii]**2 for ii in intersect])**(1/2))*(sum([neighbor[ii]**2 for ii in intersect])**(1/2)))
            if sim_den == 0:
                sim_dict[pair] = 0
                continue
            sim = sum([to_rate[ii] * neighbor[ii] for ii in intersect]) / sim_den
            # if sim < 0:
            #     sim_dict[pair] = sim
            #     continue
            if sim < 0:
                sim = sim + abs(sim) * 0.9
            else:
                sim = sim * 1.1
            sim_dict[pair] = sim
            sim_list.append(sim)
            rating_list.append(rr[1])
    # when there's no similar items: use business average
    if sim_list == []:
        other_users = list(train_RDD_bid_origin[uid_bid[0]].values())
        bid_avg = sum(other_users)/len(other_users)
        return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(bid_avg) + "\n"
    elif len(sim_list) >= 5:
        #use the X most similar items
        sim_rating = sorted(tuple(zip(sim_list, rating_list)), key=lambda x: x[0])[0:5]
        den = sum([abs(x[0]) for x in sim_rating])
        num = sum([x[0] * x[1] for x in sim_rating])
        if num <= 25:
            other_users = list(train_RDD_bid_origin[uid_bid[0]].values())
            bid_avg = sum(other_users) / len(other_users)
            # ratings = [xx[1] for xx in rated]
            # user_avg = sum(ratings) / len(ratings)
            return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(bid_avg) + "\n"
        else:
            prediction = num / den
            return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(prediction) + "\n"
    else:
        # less than three similar items: use business average
        other_users = list(train_RDD_bid_origin[uid_bid[0]].values())
        bid_avg = sum(other_users) / len(other_users)
        # ratings = [xx[1] for xx in rated]
        # user_avg = sum(ratings) / len(ratings)
        return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(bid_avg) + "\n"


# calculate predictions
sim_dict = {}
res_item = ""
res_item += test_RDD.map(lambda x: item_based(x, sim_dict)).reduce(lambda x, y: x+y)
"""
Hybrid recommender system
"""
from joblib import load
xgbc = load('model.joblib')
user_frame = user_frame.set_index("user_id").join(uid_std_df.set_index("user_id"))
business_frame = business_frame.set_index("business_id").join(bid_std_df.set_index("business_id"))
print(user_frame.head())
X_test = np.array(test_RDD.map(lambda x: get_matrix_test(x, user_frame, business_frame)).collect())
switching = xgbc.predict(X_test)
print(switching[0:20])
res = "user_id, business_id, prediction\n"
res_model_split = res_model.split("\n")
res_item_split = res_item.split("\n")

# test Vocareum
# with open("Y.csv") as in_file:
#     switching = in_file.read().split()
for i in range(len(switching)):
    if switching[i] == 1:
        res += res_model_split[i] + "\n"
    else:
        res += res_item_split[i] + "\n"
# weighted
# for i in range(len(switching)):
#     pair = ",".join(res_item_split[i].split(",")[0:2])
#     res += pair + "," + str((float(res_item_split[i].split(",")[2])+float(res_model_split[i].split(",")[2]))/2) + "\n"

with open(output_path, "w") as out_file:
    out_file.writelines(res)

"""
End timer
"""
duration = time.time() - start_time
print("Duration: "+str(duration))

# """
# See results
# """
# import numpy as np
# # calculate RMSE
# with open("task2_3.csv") as in_file:
#     guess = in_file.readlines()[1:]
# with open("data/yelp_val.csv") as in_file:
#     ans = in_file.readlines()[1:]
# res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
# RMSE = 0
# for i in range(len(guess)):
#     diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
#     RMSE += diff**2
#     if abs(diff) < 1:
#         res["<1"] = res["<1"] + 1
#     elif 2 > abs(diff) >= 1:
#         res["1~2"] = res["1~2"] + 1
#     elif 3 > abs(diff) >= 2:
#         res["2~3"] = res["2~3"] + 1
#         # if diff > 0:
#         #     large_small["large"] = large_small["large"] + 1
#         # else:
#         #     large_small["small"] = large_small["small"] + 1
#         # print(guess[i].split(","))
#         # print(ans[i].split(","))
#         # print("========")
#     elif 4 > abs(diff) >= 3:
#         res["3~4"] = res["3~4"] + 1
#     else:
#         res["4~5"] = res["4~5"] + 1
# RMSE = (RMSE/len(guess))**(1/2)
# print("RMSE: "+str(RMSE))
# prediction = np.array([float(gg.split(',')[2]) for gg in guess])
# print("Prediction mean: " + str(prediction.mean()))
# print("Prediction std:" + str(prediction.std()))
# ground = np.array([float(gg.split(',')[2]) for gg in ans])
# print("Answer mean: "+str(ground.mean()))
# print("Answer std: "+str(ground.std()))