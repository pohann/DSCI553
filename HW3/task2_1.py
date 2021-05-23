# HW3 Task2-1
# Po-Han Chen USCID: 3988044558
# Mar 03 2021

#
from pyspark import SparkContext
import os
import sys
import time
import numpy as np

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
# parse command line arguments
train_path = sys.argv[1]
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
sc = SparkContext('local[*]', 'task2_1')
sc.setLogLevel("ERROR")
# read data
# train_RDD: (business_id, user_id, stars)
# test_RDD: (business_id, user_id)
train_RDD = sc.textFile(train_path).filter(lambda x: not x.startswith("user_id")).map(
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
train_RDD = train_RDD.map(lambda x: (bid_dict[x[0]], (uid_dict[x[1]], float(x[2])))).cache()
test_RDD = test_RDD.map(lambda x: (bid_dict[x[0]], uid_dict[x[1]])).cache()
uid_len = len(uid_dict)


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
# calculate predictions
res = "user_id, business_id, prediction\n"
sim_dict = {}


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


res += test_RDD.map(lambda x: item_based(x, sim_dict)).reduce(lambda x, y: x+y)

with open(output_path, "w") as out_file:
    out_file.writelines(res)
"""
End timer
"""
duration = time.time() - start_time
print("Duration: "+str(duration))

# calculate RMSE
with open("task2_1.csv") as in_file:
    guess = in_file.readlines()[1:]
with open("data/yelp_val.csv") as in_file:
    ans = in_file.readlines()[1:]
res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
large_small = {"large": 0, "small": 0}

RMSE = 0
for i in range(len(guess)):
    # if float(guess[i].split(",")[2]) < 1:
    #     dist_guess["<1"] = dist_guess["<1"] + 1
    # elif 2 > float(guess[i].split(",")[2]) >= 1:
    #     dist_guess["1~2"] = dist_guess["1~2"] + 1
    # elif 3 > float(guess[i].split(",")[2]) >= 2:
    #     dist_guess["2~3"] = dist_guess["2~3"] + 1
    # elif 4 > float(guess[i].split(",")[2]) >= 3:
    #     dist_guess["3~4"] = dist_guess["3~4"] + 1
    # else:
    #     dist_guess["4~5"] = dist_guess["4~5"] + 1
    #
    # if float(ans[i].split(",")[2]) == 1:
    #     dist_ans["1"] = dist_ans["1"] + 1
    # elif float(ans[i].split(",")[2]) == 2:
    #     dist_ans["2"] = dist_ans["2"] + 1
    # elif float(ans[i].split(",")[2]) == 3:
    #     dist_ans["3"] = dist_ans["3"] + 1
    # elif float(ans[i].split(",")[2]) == 4:
    #     dist_ans["4"] = dist_ans["4"] + 1
    # else:
    #     dist_ans["5"] = dist_ans["5"] + 1
    diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
    RMSE += diff**2
    if abs(diff) < 1:
        res["<1"] = res["<1"] + 1
    elif 2 > abs(diff) >= 1:
        res["1~2"] = res["1~2"] + 1
        # print(guess[i].split(","))
        # print(ans[i].split(","))
        # print("========")
        # if diff > 0:
        #     large_small["large"] = large_small["large"] + 1
        # else:
        #     large_small["small"] = large_small["small"] + 1
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
# print(res)
# print(dist_ans)
# print(dist_guess)
# print(large_small)
