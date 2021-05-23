# HW3 Task1
# Po-Han Chen USCID: 3988044558
# Mar 03 2021


from pyspark import SparkContext
import os
import sys
import time
import random

import functools as fc
import itertools as it

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
# parse command line arguments
input_path = sys.argv[1]
output_path = sys.argv[2]


# read rows
def read_row(x):
    row_list = x.split(",")
    return row_list[1], row_list[0]


# create matrix of hash functions
def random_hash(b, r, distinct_users, distinct_business):
    num_hash = b * r
    hash_set = {tt for tt in tuple(zip(random.sample(range(1, sys.maxsize-1), num_hash), random.sample(range(0, sys.maxsize-1), num_hash)))}
    user_array = list(range(distinct_users))
    hash_matrix = []
    for hash_func in hash_set:
        if True:
            hash_matrix.append([((hash_func[0] * uu + hash_func[1]) % 233333333333) % (2 * distinct_business) for uu in user_array])
    return hash_matrix


# get signature
def get_sig(x, b, r, hash_matrix):
    orig_sig = []
    for hh in hash_matrix:
        orig_sig.append(min([hh[xx] for xx in x[1]]))
    res = []
    if r == 1:
        for ii in range(b):
            res.append(str(ii) + "_" + str(orig_sig[ii]))
    else:
        for ii in range(b):
            res.append(str(ii) + "_" + fc.reduce(lambda x_1, x_2: str(x_1) + ',' + str(x_2), orig_sig[ii * r:(ii + 1) * r]))
    return res


# get the buckets of candidates
def get_bucket(x, y):
    for key in y:
        val = x.get(key, set())
        if val == set():
            x[key] = y[key]
        else:
            x[key].add(list(y[key])[0])
    return x

"""
Start timer
"""
start_time = time.time()
sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")
# initialize parameters
num_b = 15
num_r = 2
# please experiment with # of partition later .partitionBy(5, lambda x: ord(x[0][0]))
RDD = sc.textFile(input_path).filter(lambda x: not x.startswith("user_id")).map(
    read_row).reduceByKey(lambda x, y: x+','+y).map(
    lambda x: (x[0], set(x[1].split(',')))).cache()
"""
user_set: all unique user_id
user_dict: all unique user_id and it's position in the vector (used for hashing)
"""
# create dictionary for user_id
user_set = RDD.reduce(lambda x, y: (1, x[1].union(y[1])))[1]
user_dict = dict(zip(user_set, range(len(user_set))))
print(len(user_set))
business_set = RDD.reduceByKey(lambda x, y: 1).collect()
# generate minhash functions
hash_matrix = random_hash(num_b, num_r, len(user_set), len(business_set))
"""
can_dict: key: string of a bucket, value: set of user_if falls into that bucket
"""
# ('NoA6bD6W7z_Aztk_cOU5cg', 'gTw6PENNGl68ZPUpYWP50A')
# get the sets of candidate pairs
can_dict = list(RDD.map(lambda x: (x[0], set(user_dict[xx] for xx in x[1]))).map(
    lambda x: (x[0], get_sig(x, num_b, num_r, hash_matrix))).map(
    lambda x: dict(zip(x[1], [set([x[0]])]*len(x[1])))).reduce(get_bucket).values())

can_dict = set(tuple(sorted(dd)) for dd in can_dict if len(dd) >= 2)
can_dict_3 = set(dd for dd in can_dict if len(dd) >= 3)
can_dict_3 = set(tuple(sorted(ddd)) for dd in can_dict_3 for ddd in it.combinations(dd, 2))
candidates = sorted(set(dd for dd in can_dict if len(dd) == 2).union(can_dict_3))
can_dict_flat = set(ddd for dd in candidates for ddd in dd)
# get candidate pairs
pairs = dict(RDD.filter(lambda x: x[0] in can_dict_flat).collect())

# write results
res = "business_id_1, business_id_2, similarity\n"
for cc in candidates:
    jaccard = len(pairs[cc[0]].intersection(pairs[cc[1]]))/len(pairs[cc[0]].union(pairs[cc[1]]))
    # only keep paris with Jaccard similarity >= 0.5
    if jaccard >= 0.5:
        res += cc[0] + "," + cc[1] + "," + str(jaccard) + "\n"
with open(output_path, "w") as out_file:
    out_file.writelines(res)
print(len(candidates))
"""
End timer
"""
duration = time.time() - start_time
print("Duration: "+str(duration))
"""
Calculate precision and recall
"""
with open("data/pure_jaccard_similarity.csv") as in_file:
    answer = in_file.read().splitlines(True)[1:]
answer_set = set()
for line in answer:
    row = line.split(',')
    answer_set.add((row[0], row[1]))
with open("task1.csv") as in_file:
    estimate = in_file.read().splitlines(True)[1:]
estimate_set = set()
for line in estimate:
    row = line.split(',')
    estimate_set.add((row[0], row[1]))
print("Precision:")
print(len(answer_set.intersection(estimate_set))/len(estimate_set))
print("Recall:")
print(len(answer_set.intersection(estimate_set))/len(answer_set))
print(answer_set.difference(estimate_set))



