"""
Method Description:
To improve the recommender system from HW3, I first decided to use model-based CF as the primary recommender for its better performance over item-based CF.
First I tried to improve model-based CF by adding more features.
One thing I found out is that feature engineering is very important.
For example, I first use one hot encoding to add "categories" of business.json.
However, there are 1000+ different categories.
As a result, adding those dummy variables directly to the model will slow down the training process and did not improve the model's performance.
So I used PCA to conduct dimensionality reduction and only kept the top 10 pca components as the features.
This resulted in about 0.002 improvement in RMSE.
On the other hand, I tried to utilize the power of item-based CF by implementing a feature-augmented CF.
I did so by adding the prediction as a feature to the model-based CF.
And that is the best model I had up to the deadline.

Other recommender systems I tried but wasn't very successful:
Use K-means to cluster the data and train model-based CF on each cluster (make predictions by first assign active user-business pair to nearest cluster)
--> One possible reason why this method is not working is because to do K-means clustering, all the points are naively represented by a point in Euclidean space
For certain type of business (mainly restaurants), collect extra features for these businesses (e.g., GoodforKids) and train two models: one for that type of businesss and one for all businesses
--> This method is not working probably because the exclusion of some valuable information from other businesses in the first model (which is about 1/4 of the training data)

Error Distribution:
>=0 and <1: 105773
>=1 and <2: 34123
>=2 and <3: 6310
>=3 and <4: 790
>=4: 0

RMSE:

Esecution Time:
around 600 seconds
"""
# Competition project
# Po-Han Chen USCID: 3988044558
# Mar 03 2021


import json
import sys
import time
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split


# read rows
def read_train(x):
    row_list = x.split(",")
    return row_list[1], row_list[0], row_list[2]


def read_test(x):
    row_list = x.split(",")
    return row_list[1], row_list[0]


def get_RMSE(test_path):
    # calculate RMSE
    with open("competition.csv") as in_file:
        guess = in_file.readlines()[1:]
    with open(test_path.replace("_in", "")) as in_file:
        ans = in_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    large_small = {"large": 0, "small": 0}

    RMSE = 0
    for i in range(len(guess)):
        diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
        RMSE += diff ** 2
        if abs(diff) < 1:
            res["<1"] = res["<1"] + 1
        elif 2 > abs(diff) >= 1:
            res["1~2"] = res["1~2"] + 1
        elif 3 > abs(diff) >= 2:
            res["2~3"] = res["2~3"] + 1
        elif 4 > abs(diff) >= 3:
            res["3~4"] = res["3~4"] + 1
        else:
            res["4~5"] = res["4~5"] + 1
    RMSE = (RMSE / len(guess)) ** (1 / 2)
    print("RMSE: " + str(RMSE))
    prediction = np.array([float(gg.split(',')[2]) for gg in guess])
    print("Prediction mean: " + str(prediction.mean()))
    print("Prediction std:" + str(prediction.std()))
    ground = np.array([float(gg.split(',')[2]) for gg in ans])
    print("Answer mean: " + str(ground.mean()))
    print("Answer std: " + str(ground.std()))


"""
Item-based CF
"""


def item_based_(uid_bid, sim_dict):
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


def item_based(uid_bid, sim_dict):
    """
    :param uid_bid: bid, uid
    :param sim_dict:
    :return: uid_bid, prediction
    """
    # if user hasn't rated anything: use business average
    # if business hasn't been rated by anyone, use 3.75
    if uid_bid[1] not in train_RDD_uid:
        if uid_bid[0] in train_RDD_bid_origin:
            other_users = list(train_RDD_bid_origin[uid_bid[0]].values())
            bid_avg = sum(other_users) / len(other_users)
            return uid_bid, bid_avg
        else:
            # return uid_dict_reversed[uid_bid[1]] + "," + bid_dict_reversed[uid_bid[0]] + "," + str(3.75) + "\n"
            return uid_bid, 3.75
    # find other items that the user have rated
    # rated: uid [(bid, stars), ...]
    # rated: uid {bid, stars, ...}
    rated = train_RDD_uid[uid_bid[1]]
    # when no other users have rated this items: use user's average rating
    if uid_bid[0] not in train_RDD_bid:
        ratings = [rated[key] for key in rated]
        user_avg = sum(ratings)/len(ratings)
        return uid_bid, user_avg
    # to_rate: item to be rated
    # to_rate: {uid: stars-mean, ...}
    to_rate = train_RDD_bid[uid_bid[0]]
    to_rate_set = set(to_rate.keys())
    # calculate prediction
    sim_list = []
    rating_list = []
    # rated: uid [(bid, stars), ...]
    # rated: uid {bid, stars, ...}
    for key in rated:
        rr = (key, rated[key]) # new line
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
        return uid_bid, bid_avg
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
            return uid_bid, bid_avg
        else:
            prediction = num / den
            return uid_bid, prediction
    else:
        # less than five similar items: use business average
        other_users = list(train_RDD_bid_origin[uid_bid[0]].values())
        bid_avg = sum(other_users) / len(other_users)
        # ratings = [xx[1] for xx in rated]
        # user_avg = sum(ratings) / len(ratings)
        return uid_bid, bid_avg


def d_mean(x):
    """
    deduct mean for each item
    :param x:
    :return:
    """
    mean = sum([xx[1] for xx in x[1]]) / len(x[1])
    res = [(xx[0], xx[1] - mean) for xx in x[1]]
    return x[0], dict([(xx[0], xx[1]) for xx in x[1]]), dict(res)


"""
Model-based CF
"""


def get_matrix_train(x, user_frame, business_frame):
    if x[1][0] in user_frame.index and x[0] in business_frame.index:
        return list(user_frame.loc[x[1][0]]) + list(business_frame.loc[x[0]]) + [res_item_train[(x[0], x[1][0])]]
    elif x[1][0] in user_frame.index and x[0] not in business_frame.index:
        return list(user_frame.loc[x[1][0]]) + [np.nan] * business_frame.shape[1] + [res_item_train[(x[0], x[1][0])]]
    elif x[1][0] not in user_frame.index and x[0] in business_frame.index:
        return [np.nan] * user_frame.shape[1] + list(business_frame.loc[x[0]]) + [res_item_train[(x[0], x[1][0])]]
    else:
        return [np.nan] * user_frame.shape[1] + [np.nan] * business_frame.shape[1] + [res_item_train[(x[0], x[1][0])]]


def get_matrix_test(x, user_frame, business_frame):
    if x[1] in user_frame.index and x[0] in business_frame.index:
        return list(user_frame.loc[x[1]]) + list(business_frame.loc[x[0]]) + [res_item_test[x]]
    elif x[1] in user_frame.index and x[0] not in business_frame.index:
        return list(user_frame.loc[x[1]]) + [np.nan] * business_frame.shape[1] + [res_item_test[x]]
    elif x[1] not in user_frame.index and x[0] in business_frame.index:
        return [np.nan] * user_frame.shape[1] + list(business_frame.loc[x[0]]) + [res_item_test[x]]
    else:
        return [np.nan] * user_frame.shape[1] + [np.nan] * business_frame.shape[1] + [res_item_test[x]]


def get_y(x):
    return x[1][1]


def read_business(x):
    return x["business_id"], x["stars"], x["review_count"], x["state"], x["categories"], x["city"], x["latitude"], x["longitude"], x["hours"]


def read_user(x):
    return x["user_id"], x["average_stars"], x["review_count"], x["fans"], x["friends"], x["yelping_since"], x["useful"], x["funny"], x["cool"], x["compliment_hot"], x["compliment_more"], x["compliment_profile"], x["compliment_cute"], x["compliment_list"], x["compliment_note"], x["compliment_plain"], x["compliment_cool"], x["compliment_funny"], x["compliment_writer"], x["compliment_photos"]


def map_business(x):
    if x[4]:
        if x[8]:
            return bid_dict[x[0]], float(x[1]), int(x[2]), x[3], set(x[4].split(", ")), x[5], float(x[6]), float(x[7]), set(x[8].keys())
        else:
            return bid_dict[x[0]], float(x[1]), int(x[2]), x[3], set(x[4].split(", ")), x[5], float(x[6]), float(x[7]), set()
    else:
        if x[8]:
            return bid_dict[x[0]], float(x[1]), int(x[2]), x[3], set(), x[5], float(x[6]), float(x[7]), set(x[8].keys())
        else:
            return bid_dict[x[0]], float(x[1]), int(x[2]), x[3], set(), x[5], float(x[6]), float(x[7]), set()


def map_user(x):
    if x[4] != "None":
        return uid_dict[x[0]], float(x[1]), int(x[2]), int(x[3])**2, len(x[4].split(","))**2, (datetime.now() - datetime.strptime(x[5], "%Y-%m-%d")).days, int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]), int(x[12]), int(x[13]), int(x[14]), int(x[15]), int(x[16]), int(x[17]), int(x[18]), int(x[19])
    else:
        return uid_dict[x[0]], float(x[1]), int(x[2]), int(x[3])**2, 0, (datetime.now() - datetime.strptime(x[5], "%Y-%m-%d")).days, int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]), int(x[12]), int(x[13]), int(x[14]), int(x[15]), int(x[16]), int(x[17]), int(x[18]), int(x[19])


if __name__ == """__main__""":
    # parse command line arguments
    folder_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    """
    Start timer
    """
    start_time = time.time()
    conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    # read data
    if 1 == 1:
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
        # train_RDD_bid: bid, {uid: stars-mean, ...}. for a bid, see who have rated this item and what's the rating
        # train_RDD_bid_origin: bid, {uid: stars, ...}. for a bid, see who have rated this item and what's the rating
        train_RDD_bid_ = train_RDD.groupByKey().map(lambda x: (x[0], list(x[1]))).map(d_mean).collect()
        train_RDD_bid = dict(zip([xx[0] for xx in train_RDD_bid_], [xx[2] for xx in train_RDD_bid_]))
        train_RDD_bid_origin = dict(zip([xx[0] for xx in train_RDD_bid_], [xx[1] for xx in train_RDD_bid_]))
        # train_RDD_uid: uid, [(bid, stars), ...] for a uid, see what items s/he have rated
        # train_RDD_uid: uid, {bid: stars, ...} for a uid, see what items s/he have rated
        # train_RDD_uid = train_RDD.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
        train_RDD_uid = train_RDD.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().map(lambda x: (x[0], list(x[1]))).map(
            lambda x: (x[0], dict(zip([xx[0] for xx in x[1]], [xx[1] for xx in x[1]])))).collect()
        train_RDD_uid = dict(zip([xx[0] for xx in train_RDD_uid], [xx[1] for xx in train_RDD_uid]))
    bool_dict = {"True": 1, "False": 0}
    noise_dict = {"quiet": 1, "average": 2, "loud": 3, "very_loud": 4}
    """
    Item-based CF
    """
    # calculate predictions
    sim_dict = {}
    res_item_test = dict()
    res_item_train = dict()
    for pair in test_RDD.collect():
        uid_bid, prediction = item_based(pair, sim_dict)
        res_item_test[uid_bid] = prediction
    # pair: bid, uid
    for pair in train_RDD.map(lambda x: (x[0], x[1][0])).collect():
        # print(pair)
        # print(train_RDD_uid[pair[1]])
        # drop predicting pair from dictionaries
        # train_RDD_uid: uid, {bid: stars, ...} for a uid, see what items s/he have rated
        # train_RDD_bid: bid, {uid: stars-mean, ...}. for a bid, see who have rated this item and what's the d-mean rating
        # train_RDD_bid_origin: bid, {uid: stars, ...}. for a bid, see who have rated this item and what's the rating
        stars_train_RDD_uid = train_RDD_uid[pair[1]].pop(pair[0])
        if train_RDD_uid[pair[1]] == dict():
            train_RDD_uid.pop(pair[1])

        stars_mean_train_RDD_bid = train_RDD_bid[pair[0]].pop(pair[1])
        if train_RDD_bid[pair[0]] == dict():
            train_RDD_bid.pop(pair[0])

        stars_train_RDD_bid_origin = train_RDD_bid_origin[pair[0]].pop(pair[1])
        if train_RDD_bid_origin[pair[0]] == dict():
            train_RDD_bid_origin.pop(pair[0])

        uid_bid, prediction = item_based(pair, sim_dict)
        res_item_train[uid_bid] = prediction
        # res_item_train += item_based(pair, sim_dict)

        # add back pair to dictionaries
        if pair[1] not in train_RDD_uid:
            train_RDD_uid[pair[1]] = dict()
            train_RDD_uid[pair[1]][pair[0]] = stars_train_RDD_uid
        else:
            train_RDD_uid[pair[1]][pair[0]] = stars_train_RDD_uid

        if pair[0] not in train_RDD_bid:
            train_RDD_bid[pair[0]] = dict()
            train_RDD_bid[pair[0]][pair[1]] = stars_mean_train_RDD_bid
        else:
            train_RDD_bid[pair[0]][pair[1]] = stars_mean_train_RDD_bid

        if pair[0] not in train_RDD_bid_origin:
            train_RDD_bid_origin[pair[0]] = dict()
            train_RDD_bid_origin[pair[0]][pair[1]] = stars_train_RDD_bid_origin
        else:
            train_RDD_bid_origin[pair[0]][pair[1]] = stars_train_RDD_bid_origin
    print("Finished Item-based prediction")

    """
    Model-based CF
    """
    # read data for model-based CF
    if 1 == 1:
        # read user.json
        user_data = []
        with open(folder_path + "user.json") as in_file:
            while True:
                data = in_file.readlines(5000000)
                if not data:
                    break
                user_RDD = sc.parallelize(data).map(lambda x: json.loads(x)).map(read_user).filter(lambda x: x[0] in uid_dict).map(map_user)
                user_data += user_RDD.collect()
        # read business.json
        business_RDD = sc.textFile(folder_path+"business.json").map(lambda x: json.loads(x)).map(read_business).filter(lambda x: x[0] in bid_dict).map(
            map_business)
        business_data = business_RDD.collect()

        # preprocessing data with pandas
        user_frame = pd.DataFrame(user_data, columns=["user_id", "average_stars", "review_count", "fans_sqr", "friends_sqr", "yelping_since", "useful", "funny", "cool", "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"])
        business_frame = pd.DataFrame(business_data, columns=["business_id", "stars", "review_count", "state", "categories", "city", "latitude", "longitude", "hours"])
        # get pca for compliment
        compliment = np.array(user_frame.iloc[:, 9:])
        print(compliment.shape)
        pca = PCA(n_components=5, svd_solver='full')
        compliment = pca.fit_transform(compliment)
        print(compliment.shape)
        compliment = np.transpose(compliment)
        for i in range(len(compliment)):
            user_frame["compliment_"+str(i)] = compliment[i]
        user_frame = user_frame.drop(columns=["compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"])


        # get dummies of city
        city = pd.get_dummies(business_frame.city, prefix="city")
        city = np.array(city)
        # city = np.transpose(np.transpose(city)[:-1])
        pca = PCA(n_components=10, svd_solver='full')
        city = pca.fit_transform(city)
        city = np.transpose(city)
        for i in range(len(city)):
            business_frame["city_"+str(i)] = city[i]
        business_frame = business_frame.drop(columns=["city"])
        # get dummies of state
        state = pd.get_dummies(business_frame.state, prefix="state")
        state = state.drop(columns=[state.columns[-1]])
        business_frame = pd.concat([business_frame, state], axis=1)
        business_frame = business_frame.drop(columns=["state"])
        # get dummies of categories
        category_set = set()
        for cc in business_frame.categories:
            category_set = set(chain(category_set, cc))
        category_matrix = []
        for cc in category_set:
            category_matrix.append(business_frame.categories.apply(lambda x: x.intersection(set([cc]))).apply(lambda x: 0 if x == set() else 1))
        category_matrix = np.transpose(np.array(category_matrix))
        pca = PCA(n_components=10, svd_solver='full')
        category_matrix = pca.fit_transform(category_matrix)
        category_matrix = np.transpose(category_matrix)
        for i in range(len(category_matrix)):
            business_frame["category_"+str(i)] = category_matrix[i]
        business_frame = business_frame.drop(columns=["categories"])
        # get dummies of hours
        print(list(business_frame.hours)[0:5])
        hours_set = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hours_matrix = []
        for hh in hours_set:
            hours_matrix.append(business_frame.hours.apply(lambda x: x.intersection(set([hh]))).apply(lambda x: 0 if x == set() else 1))
        for i in range(len(hours_matrix)):
            business_frame["Day_"+str(i)] = hours_matrix[i]
        business_frame = business_frame.drop(columns=["hours"])
        print(list(business_frame.Day_1)[0:10])


    print("Finished reading files for model-based CF")
    # single model
    user_frame_pred = user_frame.set_index("user_id")
    business_frame_pred = business_frame.set_index("business_id")
    X = []
    for tt in train_RDD.collect():
        X.append(get_matrix_train(tt, user_frame_pred, business_frame_pred))
    X = np.array(X)
    X_test = []
    for tt in test_RDD.collect():
        X_test.append(get_matrix_test(tt, user_frame_pred, business_frame_pred))
    X_test = np.array(X_test)
    Y = np.array(train_RDD.map(get_y).collect())

    print(X.shape)
    print("Finished preparing X and Y")

    duration = time.time() - start_time
    print("Duration: "+str(duration))


    """
    XGBRegressor-Single model
    """
    import xgboost as xgb
    from joblib import dump, load

    xgbr = load("model.md")
    # make predictions
    Y_pred = xgbr.predict(X_test)
    # write results
    res_model = ""
    rows = test_RDD.collect()
    for i in range(len(rows)):
        if Y_pred[i] < 1:
            res_model += uid_dict_reversed[rows[i][1]] + "," + bid_dict_reversed[rows[i][0]] + "," + str(1) + "\n"
        elif Y_pred[i] > 5:
            res_model += uid_dict_reversed[rows[i][1]] + "," + bid_dict_reversed[rows[i][0]] + "," + str(5) + "\n"
        else:
            res_model += uid_dict_reversed[rows[i][1]] + "," + bid_dict_reversed[rows[i][0]] + "," + str(
                Y_pred[i]) + "\n"


    """
    Final output
    """
    res = "user_id, business_id, stars\n"
    res += res_model
    # write file
    with open(output_path, "w") as out_file:
        out_file.writelines(res)
    get_RMSE(test_path)
    """
    End timer
    """
    duration = time.time() - start_time
    print("Duration: "+str(duration))



