# HW4 Task2
# Po-Han Chen USCID: 3988044558
# Mar 25 2021


from pyspark import SparkContext, SparkConf
import os
import sys
import time
import itertools
from functools import reduce

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def check(x, previous):
    return x[0], set(xx for xx in x[1] if xx not in previous[x[0][0]])


def BFS(bu_joined, depth, shortest_path, direct_dict, previous):
    """
    Performs BFS on the graph to find all shortest path
    :param bu_RDD:
    :return:
    """
    # print("current depth: " + str(depth))
    current = dict(bu_joined.map(lambda x: (x[0], set(x))).reduceByKey(lambda x, y: x.union(y)).collect())
    for key in current:
        previous[key] = previous.get(key, set()).union(current[key])
    bu_joined = bu_joined.map(lambda x: (x, direct_dict[x[-1]])).map(
        lambda x: check(x, previous)).cache()
    # bu_joined = bu_joined.map(lambda x: (x, direct_dict[x[-1]])).map(
    #     lambda x: (x[0], set(xx for xx in x[1] if xx not in x[0]))).map(
    #     lambda x: (x[0], set(xx for xx in x[1] if direct_dict[xx].intersection(set(x[0][:-1])) == set()))).cache()
    # if x[1] == set(), we've found a shortest path!

    # shortest_path = shortest_path.union(set(bu_joined.filter(lambda x: x[1] == set()).map(lambda x: x[0]).collect()))
    shortest_path[depth] = tuple(set(bu_joined.filter(lambda x: x[1] == set()).map(lambda x: x[0]).collect()))
    # filter out shortest path
    bu_joined = bu_joined.filter(lambda x: x[1] != set()).cache()
    if bu_joined.count() == 0:
        res = set()
        for key in shortest_path:
            for pp in shortest_path[key]:
                res.add(pp)
        return res, depth
    # prepared for next iteration
    bu_joined = bu_joined.flatMap(
        lambda x: [x[0] + tuple([xx]) for xx in x[1]]).cache()
    return BFS(bu_joined, depth+1, shortest_path, direct_dict, previous)


def graph_builder(x):
    """
    BUild the graphs based on the shortest paths
    :param x:
    :return:
    """

    # tree: first element: {root:[]}
    # each layer is a dictionary where
    # key = node, values = set(parents)
    tree = [{x[0]:0}]
    # node credit
    # keep track of the credit on each node
    node_credit = {x[0]: 0}
    # edge credit
    # keep track of the edge credit
    edge_credit = dict()
    # number of shortest paths that pass through each node
    num_path = {x[0]: 1}
    # keep track of shortest paths passing through parents to child
    edge_weight = dict()
    if x[0] != x[1][0]:
        # build tree
        for pp in x[1]:
            for i in range(len(pp[1:])):
                if len(tree) < i + 2:
                    tree.append(dict())
                tree[i+1][pp[i+1]] = tree[i+1].get(pp[i+1], set()).union(set([pp[i]]))
                edge_weight[tuple(sorted(pp[i:i+2]))] = edge_weight.get(tuple(sorted(pp[i:i+2])), 0) + 1

        # calculate credit
        for i in range(len(tree)-1):
            for key in tree[len(tree) - i - 1]:
                parents = [xx for xx in tree[len(tree) - i - 1][key]]
                total_paths = sum([edge_weight[tuple(sorted([xx, key]))] for xx in parents])
                if key not in node_credit:
                    node_credit[key] = 1
                c = node_credit[key]
                for pp in parents:
                    edge_credit[tuple(sorted([key, pp]))] = edge_credit.get(tuple(sorted([key, pp])), 0) + (edge_weight[tuple(sorted([key, pp]))]/total_paths) * c
                    node_credit[pp] = node_credit.get(pp, 1) + (edge_weight[tuple(sorted([key, pp]))]/total_paths) * c
        return tree, num_path, node_credit, edge_credit
    else:
        return tree, num_path, node_credit, edge_credit


def deal_single_node(x):
    if x[1] == set():
        return (x[0], x[0])
    else:
        return [(x[0], xx) for xx in x[1]]


def Q(bu_joined_origin, direct_dict, A, m, K, previsou_Q, previous_community_set, previous=dict(), shortest_path=dict()):
    """
    Find community
    :param bu_joined:
    :param depth:
    :param shortest_path:
    :param direct_dict:
    :param previous:
    :return:
    """
    # first find the shortest paths based on the new direct_dict
    bu_joined = bu_joined_origin.flatMap(lambda x: [x, (x[1], x[0])]).reduceByKey(lambda x, y: 0).map(lambda x: x[0])
    # first iteration
    bu_joined = bu_joined.map(lambda x: (x, direct_dict[x])).flatMap(
        deal_single_node).cache()
    # rest of the iterations
    shortest_path, max_depth = BFS(bu_joined, 2, shortest_path, direct_dict, previous)
    # find communities
    path_RDD = sc.parallelize(shortest_path).map(lambda x: (x[0], x)).groupByKey().cache()
    communities = path_RDD.map(
        lambda x: (x[0], [xx for xx in x[1]])).map(lambda x: set(xxx for xx in x[1] for xxx in xx)).collect()
    community_set = set()
    for cc in communities:
        community_set.add(tuple(sorted(cc)))
    if previous_community_set == community_set:
        return previsou_Q, path_RDD, previous_community_set
    else:
        Q = 0
        for cc in community_set:
            if len(cc) > 1:
                for pair in itertools.combinations(cc, 2):
                    pair = tuple(sorted(pair))
                    Q += ((A[pair] - K[pair[0]] * K[pair[1]] / (2.0 * m)) / (2 * m))*2
                for node in cc:
                    pair = (node, node)
                    Q += (A[pair] - K[pair[0]] * K[pair[1]] / (2.0 * m)) / (2 * m)
            else:
                continue
        return Q, path_RDD, community_set


if __name__ == "__main__":
    # parse command line arguments
    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    b_output_path = sys.argv[3]
    c_output_path = sys.argv[4]

    """
    Start timer
    """
    start_time = time.time()

    conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # ub_RDD: [uid, bid]
    ub_RDD = sc.textFile(input_path).filter(lambda x: not x.startswith("user_id")).map(lambda x: x.split(',')).cache()
    # uid_dict: {uid:num, ... }
    uid_dict_ = ub_RDD.map(lambda x: x[0]).distinct().collect()
    uid_dict = dict(zip(uid_dict_, range(len(uid_dict_))))
    uid_dict_reversed = dict(zip(range(len(uid_dict_)), uid_dict_))
    # bid_dict: {bid:[num], ... }
    bid_dict = ub_RDD.map(lambda x: x[1]).distinct().collect()
    bid_dict = dict(zip(bid_dict, range(len(bid_dict))))
    # bu_RDD: (bid, uid)
    bu_RDD = ub_RDD.map(lambda x: (uid_dict[x[0]], bid_dict[x[1]])).map(lambda x: (x[1], x[0]))
    # bu_joined: (uid1,uid2) pairs of uid that has more than 7 co-rated items
    # assume there's no duplicate in the original file of uid, bid pairs
    # remember to filter out pairs where uid1 == uid2
    # x[1]/2 to get the real count of co-rated bid
    bu_joined_origin = bu_RDD.join(bu_RDD).map(lambda x: x[1]).filter(lambda x: x[0] != x[1]).map(
        lambda x: (tuple(sorted(x)), 1)).reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], x[1] / 2)).filter(
        lambda x: x[1] >= threshold).map(lambda x: x[0]).cache()
    # create dictionary to store betweenness values
    between_dict = dict(zip(bu_joined_origin.collect(), [0] * bu_joined_origin.count()))

    """
    Compute betweenness: Step 1. Get all the shortest paths
    """
    # bu_joined didn't have duplicates:
    # if uid1 and uid2 are connected, there will only be either (uid1,uid2) or (uid2,uid1) in the RDD
    # But to find betweeness, we need to find shortest paths starting from EACH uid
    # Thus, here we first make the copy for each pair in which the uid are switched

    # direct_dict: uid: {uid with direct connections}
    # bu_joined: uid1, uid2, ....
    direct_dict = dict(bu_joined_origin.flatMap(lambda x: [x, (x[1], x[0])]).map(lambda x: (x[0], set([x[1]]))).reduceByKey(
        lambda x, y: x.union(y)).collect())

    bu_joined = bu_joined_origin.flatMap(lambda x: [x, (x[1], x[0])]).reduceByKey(lambda x, y: 0).map(lambda x: x[0])
    shortest_path = dict()
    # """
    # Method 2: build the tree directly
    # """
    # edge_credits = bu_joined.map(lambda x: graph_builder_2(x, direct_dict)).map(lambda x: x[3]).collect()

    # when appending new node to the queue, check :
    # if the appended node is already in the tree
    # first iteration
    bu_joined = bu_joined.map(lambda x: (x, direct_dict[x])).flatMap(
        lambda x: [(x[0], xx) for xx in x[1]]).cache()
    previous = dict()
    # rest of the iterations
    shortest_path, max_depth = BFS(bu_joined, 2, shortest_path, direct_dict, previous)

    print(len(shortest_path))

    """
    Compute betweenness: Step 2. Build the graphs and get edge credit for all edges
    """
    # paths_RDD: root, [shortest paths, ...]
    paths_RDD = sc.parallelize(shortest_path).map(lambda x: (x[0], x)).groupByKey().map(
        lambda x: (x[0], [xx for xx in x[1]])).map(lambda x: graph_builder(x)).map(lambda x: x[3])
    edge_credits = paths_RDD.collect()
    """
    Compute betweenness: Step 3. Calculate betweenness
    """
    for cc in edge_credits:
        for key in cc:
            between_dict[key] += cc[key]/2.0
    # write file
    res_data = [sorted([uid_dict_reversed[key[0]], uid_dict_reversed[key[1]]]) + [between_dict[key]] for key in between_dict]
    res_data = sorted(res_data, key=lambda x: (-x[2], x[0], x[1]))
    res = ""

    for i in range(len(res_data)):
        res += "('" + res_data[i][0] + "', '" + res_data[i][1] + "')," + str(round(res_data[i][2], 5)) + "\n"

    with open(b_output_path, "w") as out_file:
        out_file.writelines(res)
    """
    Finding communities with Q modularity
    """
    # get m, A, K of the original graph
    m = bu_joined_origin.count()
    print(m)
    A = dict()
    nodes = set(bu_joined_origin.flatMap(lambda x: [x, (x[1], x[0])]).reduceByKey(lambda x, y: 0).map(lambda x: x[0]).collect())
    print(len(nodes))
    for pair in itertools.combinations(nodes, 2):
        pair = tuple(sorted(pair))
        if pair[1] in direct_dict[pair[0]]:
            A[pair] = 1
        else:
            A[pair] = 0
    for node in nodes:
        A[(node, node)] = 0
    K = dict()
    for node in direct_dict:
        K[node] = len(direct_dict[node])

    communities = sc.parallelize(shortest_path).map(lambda x: (x[0], x)).groupByKey().map(
        lambda x: (x[0], [xx for xx in x[1]])).map(lambda x: set(xxx for xx in x[1] for xxx in xx)).collect()
    previous_community_set = set()
    for cc in communities:
        previous_community_set.add(tuple(sorted(cc)))
    previous_Q = 0
    for cc in previous_community_set:
        if len(cc) > 1:
            for pair in itertools.combinations(cc, 2):
                pair = tuple(sorted(pair))
                previous_Q += ((A[pair] - K[pair[0]]*K[pair[1]]/(2.0 * m))/(2 * m)) * 2
            for node in cc:
                pair = (node, node)
                previous_Q += (A[pair] - K[pair[0]] * K[pair[1]] / (2.0 * m)) / (2 * m)
        else:
            continue
    print(previous_Q)
    while True:
        # break the connection with highest betweenness
        between_data = [(key, between_dict[key]) for key in between_dict]
        between_data = sorted(between_data, key=lambda x: -x[1])
        highest_pair = between_data[0]
        for pair in between_data:
            if pair[1] == highest_pair[1]:
                direct_dict[pair[0][0]].remove(pair[0][1])
                direct_dict[pair[0][1]].remove(pair[0][0])
        # find the shortest paths and compute Q again
        current_Q, path_RDD, current_community_set = Q(bu_joined_origin, direct_dict, A, m, K, previous_Q, previous_community_set, previous=dict(), shortest_path=dict())
        print(current_Q)
        if current_Q < previous_Q:
            break
        previous_Q = current_Q
        previous_community_set = current_community_set
        edge_credits = path_RDD.map(lambda x: (x[0], [xx for xx in x[1]])).map(lambda x: graph_builder(x)).map(lambda x: x[3]).collect()
        for key in between_dict:
            between_dict[key] = 0
        for cc in edge_credits:
            for key in cc:
                between_dict[key] += cc[key] / 2.0
    res_community = []
    for cc in previous_community_set:
        res_community.append(sorted(["'"+uid_dict_reversed[node]+"'" for node in cc]))
    res_community = sorted(res_community, key=lambda x: (len(x), x[0]))

    res = ""
    for rc in res_community:
        res += reduce(lambda x, y: x+", "+y, rc)
        res += "\n"

    with open(c_output_path, "w") as out_file:
        out_file.writelines(res)
    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: " + str(duration))

