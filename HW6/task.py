# HW6
# Po-Han Chen USCID: 3988044558
# Apr 13 2021


import sys
import time
import random
from sklearn.cluster import KMeans
import numpy as np
import copy
import itertools


def m_distance_point(point, cluster):
    """
    Calculate Mahalanobis distance between a point and a cluster
    :param point:
    :param cluster: dictionary of a cluster
    :return:
    """
    centroid = cluster["SUM"] / len(cluster["N"])
    sigma = cluster["SUMSQ"] / len(cluster["N"]) - (cluster["SUM"]/len(cluster["N"]))**2
    z = (point - centroid)/sigma
    m_distance = np.dot(z, z) ** (1/2)
    return m_distance


def m_distance_cluster(cluster_1, cluster_2):
    """
    Calculate Mahalanobis distance between two clusters
    :param cluster_1:
    :param cluster_2:
    :return:
    """
    centroid_1 = cluster_1["SUM"] / len(cluster_1["N"])
    centroid_2 = cluster_2["SUM"] / len(cluster_2["N"])
    sig_1 = cluster_1["SUMSQ"] / len(cluster_1["N"]) - (cluster_1["SUM"] / len(cluster_1["N"]))**2
    sig_2 = cluster_2["SUMSQ"] / len(cluster_2["N"]) - (cluster_2["SUM"] / len(cluster_2["N"]))**2
    z_1 = (centroid_1 - centroid_2) / sig_1
    z_2 = (centroid_1 - centroid_2) / sig_2
    m_1 = np.dot(z_1, z_1) ** (1/2)
    m_2 = np.dot(z_2, z_2) ** (1/2)
    return min(m_1, m_2)


def round_result(DS, CS, RS):
    DS_total = 0
    for cluster in DS:
        DS_total += len(DS[cluster]["N"])
    CS_cluster = len(CS)
    CS_total = 0
    for cluster in CS:
        CS_total += len(CS[cluster]["N"])
    RS_total = len(RS)
    return DS_total, CS_cluster, CS_total, RS_total


if __name__ == "__main__":
    # parse command line arguments
    input_path = sys.argv[1]
    num_cluster = int(sys.argv[2])
    output_path = sys.argv[3]

    """
    Start timer
    """
    start_time = time.time()

    """
    BFR Algorithm
    """
    res = "The intermediate results:\n"
    if 0 == 0:
        # keep track of outliers
        RS = []
        # load data
        with open(input_path) as in_file:
            data = in_file.readlines()
        data = list(map(lambda x: x.strip("\n").split(','), data))
        # data: (id, vector)
        data = [(int(dd[0]), tuple(list(map(lambda x:float(x), dd[2:])))) for dd in data]
        # data_dict: {id:vector}
        data_dict = dict(data)
        print(len(data_dict))
        """
        Add code here to deal wit duplicate points (identical vector)
        """
        # data_dict_reversed: {vector(tuple):id}
        data_dict_reversed = dict(zip(list(data_dict.values()), list(data_dict.keys())))
        # covert vector to numpy array
        data = list(map(lambda x: np.array(x), list(data_dict.values())))
        # randomly shuffle the list to simulate random sampling
        random.shuffle(data)
        chunk_len = round(len(data)/5)

    # step 1: load 20% of the data randomly
    chunk_data = data[0:chunk_len]

    # step 2: run K-means with large K
    k_means = KMeans(n_clusters=num_cluster*25).fit(chunk_data)

    # step 3: move clusters with only one points to RS
    if 3 == 3:
        # cluster_dict: {cluster id: # of points in the cluster}
        cluster_dict = dict()
        for label in k_means.labels_:
            cluster_dict[label] = cluster_dict.get(label, 0) + 1
        # find the index of the RS points
        RS_index = []
        for key in cluster_dict:
            if cluster_dict[key] < 20:
                RS_index += [i for i, x in enumerate(k_means.labels_) if x == key]
        # print(RS_key)
        # # find the index of the RS points
        # RS_index = []
        # if RS_key:
        #     for key in RS_key:
        #         RS_index.append(list(k_means.labels_).index(key))
        # append those points to RS
        for index in RS_index:
            RS.append(chunk_data[index])
        # delete points from chunk_data
        for index in reversed(sorted(RS_index)):
            chunk_data.pop(index)

    # step 4: run k-means with k clusters on data points not in RS
    k_means = KMeans(n_clusters=num_cluster).fit(chunk_data)

    # step 5: generate DS clusters
    if 5 == 5:
        # cluster pair: (cluster id: index of point)
        cluster_pair = tuple(zip(k_means.labels_, chunk_data))
        # DS: {cluster id:{N:[id], SUM:, SUMSQ:}}
        DS = dict()
        for pair in cluster_pair:
            if pair[0] not in DS:
                DS[pair[0]] = dict()
                DS[pair[0]]["N"] = [data_dict_reversed[tuple(pair[1])]]
                DS[pair[0]]["SUM"] = pair[1]
                DS[pair[0]]["SUMSQ"] = pair[1]**2
            else:
                DS[pair[0]]["N"].append(data_dict_reversed[tuple(pair[1])])
                DS[pair[0]]["SUM"] += pair[1]
                DS[pair[0]]["SUMSQ"] += pair[1]**2

    # step 6: run k-means on RS to generate CS and RS
    if RS:
        if len(RS) > 1:
            k_means = KMeans(n_clusters=len(RS) - 1).fit(RS)
        else:
            k_means = KMeans(n_clusters=len(RS)).fit(RS)
        # cluster_dict: {cluster id: # of points in the cluster}
        cluster_dict = dict()
        for label in k_means.labels_:
            cluster_dict[label] = cluster_dict.get(label, 0) + 1
        # RS_key: index of cluster that only has one point in it
        RS_key = []
        for key in cluster_dict:
            if cluster_dict[key] == 1:
                RS_key.append(key)
        # RS_index: index of the RS points in the dataset
        RS_index = []
        if RS_key:
            for key in RS_key:
                RS_index.append(list(k_means.labels_).index(key))
        # add points from RS to CS
        # cluster pair: (cluster id: index of point)
        cluster_pair = tuple(zip(k_means.labels_, RS))
        # CS: {cluster id:{N:, SUM:, SUMSQ:}}
        CS = dict()
        for pair in cluster_pair:
            if pair[0] not in RS_key:
                if pair[0] not in CS:
                    CS[pair[0]] = dict()
                    CS[pair[0]]["N"] = [data_dict_reversed[tuple(pair[1])]]
                    CS[pair[0]]["SUM"] = pair[1]
                    CS[pair[0]]["SUMSQ"] = pair[1]**2
                else:
                    CS[pair[0]]["N"].append(data_dict_reversed[tuple(pair[1])])
                    CS[pair[0]]["SUM"] += pair[1]
                    CS[pair[0]]["SUMSQ"] += pair[1]**2
        # update RS (remove points added to CS)
        new_RS = []
        for index in reversed(sorted(RS_index)):
            new_RS.append(RS[index])
        RS = copy.deepcopy(new_RS)

    DS_total, CS_cluster, CS_total, RS_total = round_result(DS, CS, RS)
    res += "Round 1: " + str(DS_total) + "," + str(CS_cluster) + "," + str(CS_total) + "," + str(RS_total) + "\n"

    for _ in range(0, 4):
        # step 7: load another 20% of data
        if _ == 3:
            chunk_data = data[chunk_len*4:]
        else:
            chunk_data = data[chunk_len*(_+1):chunk_len*(_+2)]

        # step 8: assign points to DS when they're close enough to the centroid
        if 8 == 8:
            # keep track of points that's already assigned to DS
            DS_index = set()
            for i in range(len(chunk_data)):
                point = chunk_data[i]
                distance_dict = dict()
                for cluster in DS:
                    distance_dict[cluster] = m_distance_point(point, DS[cluster])
                m_distance = min(list(distance_dict.values()))
                for cc in distance_dict:
                    if distance_dict[cc] == m_distance:
                        cluster = cc
                if m_distance < 2 * (len(point) ** (1/2)):
                    # add point to DS
                    DS[cluster]["N"].append(data_dict_reversed[tuple(point)])
                    DS[cluster]["SUM"] += point
                    DS[cluster]["SUMSQ"] += point ** 2
                    # add index to DS_index
                    DS_index.add(i)

        # step 9: assign remaining points to CS when they're close enough to the centroid
        if CS:
            # keep track of points that's already assigned to CS
            CS_index = set()
            for i in range(len(chunk_data)):
                if i not in DS_index:
                    point = chunk_data[i]
                    distance_dict = dict()
                    for cluster in CS:
                        distance_dict[cluster] = m_distance_point(point, CS[cluster])
                    m_distance = min(list(distance_dict.values()))
                    for cc in distance_dict:
                        if distance_dict[cc] == m_distance:
                            cluster = cc
                    if m_distance < 2 * (len(point) ** (1 / 2)):
                        # add point to CS
                        CS[cluster]["N"].append(data_dict_reversed[tuple(point)])
                        CS[cluster]["SUM"] += point
                        CS[cluster]["SUMSQ"] += point ** 2
                        # add index to DS_index
                        CS_index.add(i)

        # step 10: assign points not in CS and DS to RS
        if 10 == 10:
            try:
                # index of points in DS and CS
                all_index = CS_index.union(DS_index)
            except NameError:
                all_index = DS_index
            for i in range(len(chunk_data)):
                if i not in all_index:
                    RS.append(chunk_data[i])

        # step 11: run k-means on RS to update CS and RS
        if RS:
            if len(RS) > 1:
                k_means = KMeans(n_clusters=len(RS) - 1).fit(RS)
            else:
                k_means = KMeans(n_clusters=len(RS)).fit(RS)
            # to avoid adding points to cluster already exists in CS, must change the duplicate labels
            CS_cluster_set = set(CS.keys())
            RS_cluster_set = set(k_means.labels_)
            intersection = CS_cluster_set.intersection(RS_cluster_set)
            union = CS_cluster_set.union(RS_cluster_set)
            change_dict = dict()
            for ii in intersection:
                while True:
                    random_int = random.randint(100, len(chunk_data))
                    if random_int not in union:
                        break
                change_dict[ii] = random_int
                union.add(random_int)
            # get the new k-means labels
            labels = list(k_means.labels_)
            for i in range(len(labels)):
                if labels[i] in change_dict:
                    labels[i] = change_dict[labels[i]]
            # cluster_dict: {cluster id: # of points in the cluster}
            cluster_dict = dict()
            for label in labels:
                cluster_dict[label] = cluster_dict.get(label, 0) + 1
            # RS_key: index of cluster that only has one point in it
            RS_key = []
            for key in cluster_dict:
                if cluster_dict[key] == 1:
                    RS_key.append(key)
            # RS_index: index of the RS points in the dataset
            RS_index = []
            if RS_key:
                for key in RS_key:
                    RS_index.append(labels.index(key))
            # add points from RS to CS
            # cluster pair: (cluster id: index of point)
            cluster_pair = tuple(zip(labels, RS))
            for pair in cluster_pair:
                if pair[0] not in RS_key:
                    if pair[0] not in CS:
                        CS[pair[0]] = dict()
                        CS[pair[0]]["N"] = [data_dict_reversed[tuple(pair[1])]]
                        CS[pair[0]]["SUM"] = pair[1]
                        CS[pair[0]]["SUMSQ"] = pair[1] ** 2
                    else:
                        CS[pair[0]]["N"].append(data_dict_reversed[tuple(pair[1])])
                        CS[pair[0]]["SUM"] += pair[1]
                        CS[pair[0]]["SUMSQ"] += pair[1] ** 2
            # update RS (remove points added to CS)
            new_RS = []
            for index in reversed(sorted(RS_index)):
                new_RS.append(RS[index])
            RS = copy.deepcopy(new_RS)

        # step 12: merge CS clusters
        flag = True
        while True:
            compare_list = list(itertools.combinations(list(CS.keys()), 2))
            original_cluster = set(CS.keys())
            merge_list = []
            for compare in compare_list:
                m_distance = m_distance_cluster(CS[compare[0]], CS[compare[1]])
                if m_distance < 2 * (len(CS[compare[0]]["SUM"]) ** (1/2)):
                    CS[compare[0]]["N"] = CS[compare[0]]["N"] + CS[compare[1]]["N"]
                    CS[compare[0]]["SUM"] += CS[compare[1]]["SUM"]
                    CS[compare[0]]["SUMSQ"] += CS[compare[1]]["SUMSQ"]
                    CS.pop(compare[1])
                    flag = False
                    break
            new_cluster = set(CS.keys())
            # no clusters to merge
            if new_cluster == original_cluster:
                break

        # last round: merge CS to DS if they're close enough
        CS_cluster = list(CS.keys())
        if _ == 3 and CS:
            for cluster_cs in CS_cluster:
                distance_dict = dict()
                for cluster_ds in DS:
                    distance_dict[cluster_ds] = m_distance_cluster(DS[cluster_ds], CS[cluster_cs])
                m_distance = min(list(distance_dict.values()))
                for cc in distance_dict:
                    if distance_dict[cc] == m_distance:
                        cluster = cc
                if m_distance < 2 * len(CS[cluster_cs]["SUM"]) ** (1/2):
                    DS[cluster]["N"] = DS[cluster]["N"] + CS[cluster_cs]["N"]
                    DS[cluster]["SUM"] += CS[cluster_cs]["SUM"]
                    DS[cluster]["SUMSQ"] += CS[cluster_cs]["SUMSQ"]
                    CS.pop(cluster_cs)
        DS_total, CS_cluster, CS_total, RS_total = round_result(DS, CS, RS)
        res += "Round " + str(_+2) + ": " + str(DS_total) + "," + str(CS_cluster) + "," + str(CS_total) + "," + str(RS_total) + "\n"

    res += "\nThe clustering results:\n"
    for cluster in DS:
        DS[cluster]["N"] = set(DS[cluster]["N"])
    if CS:
        for cluster in CS:
            CS[cluster]["N"] = set(CS[cluster]["N"])

    RS_set = set()
    for point in RS:
        RS_set.add(data_dict_reversed[tuple(point)])

    for point in range(len(data_dict)):
        if point in RS_set:
            res += str(point) + ",-1\n"
        else:
            for cluster in DS:
                if point in DS[cluster]["N"]:
                    res += str(point) + "," + str(cluster) + "\n"
                    break
            for cluster in CS:
                if point in CS[cluster]["N"]:
                    res += str(point) + ",-1\n"
                    break

    with open(output_path, "w") as out_file:
        out_file.writelines(res)

    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: " + str(duration))
