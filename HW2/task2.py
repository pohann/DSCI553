# HW2 task2
# Po-Han Chen USC ID:

from pyspark import SparkContext
import os
import sys
import time

import functools as fc
import itertools as it

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
# parse command line arguments
threshold = sys.argv[1]
support = sys.argv[2]
input_path = sys.argv[3]
output_path = sys.argv[4]
"""
Define useful functions to help Spark implementation
"""


# finding candidates
def add_size(iterator):
    iter_list = list(iterator)
    return [(ii[1], len(iter_list)) for ii in iter_list]


def get_kvp_c(x):
    return list(zip(list(x[0]), [1/x[1]] * len(x[0])))


def get_dict(x, y):
    x[y[0]] = x.get(y[0], 0) + y[1]
    return x


def count_par(iterator):
    return [fc.reduce(get_dict, iterator, {})]


# verifying candidates
def get_kvp_v(x):
    return list(zip(list(x[1]), [1] * len(x[1])))


def printf(iterator):
    print(list(iterator))
    print("="*20)


# getting new basket
def flat_set(x):
    if pass_num == 2:
        return x
    else:
        return x[0], set(xxx for xx in x[1] for xxx in xx)


# reading data
def read_row(x):
    row_data = x.split(',')
    return row_data[0].strip('"')[0:-4]+row_data[0].strip('"')[-2:]+'-'+row_data[1].strip('"').lstrip("0"), row_data[5].strip('"').lstrip("0")


"""
Start timer
"""
start_time = time.time()
sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")
# drop the header
with open(input_path) as in_file:
    data = in_file.read().splitlines(True)[1:]
RDD = sc.parallelize(data).map(read_row)
res = RDD.collect()
# write pre-processed data
with open('ta_feng_new.csv', 'w') as out_file:
    out_file.write("DATE-CUSTOMER_ID1,PRODUCT_ID1\n")
    res = [rr[0]+','+rr[1]+'\n' for rr in res]
    out_file.writelines(res)
# read the preprocessed data
with open('ta_feng_new.csv') as in_file:
    data = in_file.read().splitlines(True)[1:]
RDD = sc.parallelize(data)

# keep track of pass
pass_num = 1
# initialize res string to write to file
res = ''
res_can = ''
res_freq = ''
# map to key-value pairs
RDD = RDD.map(lambda x: tuple(x.strip().split(','))).partitionBy(2, lambda x: ord(x[0][0]))
# get baskets: user_id:[business_id, ...] and filter data according to threshold
original_RDD = RDD.reduceByKey(lambda x, y: x+','+y).map(lambda x: (x[0], set(x[1].split(',')))).filter(lambda x: len(x[1]) > float(threshold)).cache()
support_ratio = float(support)/RDD.count()
# a-priori
while True:
    if pass_num == 1:
        basket_RDD = original_RDD
    # starting from pass 2, create new basket with size = pass_num itemsets
    elif pass_num == 2:
        # delete non frequent items from basket and create new baskets
        basket_RDD = original_RDD.map(lambda x: (x[0], x[1].intersection(freq_sets))).map(
            lambda x: (x[0], {tuple(sorted(xx)) for xx in set(it.combinations(x[1], pass_num))}) if len(
                x[1]) >= pass_num else (x[0], set()))
    # get candidates
    candidates = basket_RDD.mapPartitions(add_size).filter(lambda x: x[0] != set()).flatMap(
        get_kvp_c).mapPartitions(count_par).flatMap(
        lambda x: [(k, 1) for k, v in x.items() if v >= support_ratio]).reduceByKey(lambda x, y: 1).collect()
    """
    Needs rework on sorting the tuples (keys)
    """
    candidates = sorted(candidates)
    # if candidates is empty: break loop
    if not candidates:
        break
    for can in candidates:
        if pass_num == 1:
            res_can += "('"+can[0]+"'),"
        else:
            res_can += str(tuple(sorted(can[0])))+","
    res_can = res_can.rstrip(',') + '\n\n'
    candidates = set([can[0] for can in candidates])
    # verify candidates
    freq_sets = basket_RDD.map(lambda x: (x[0], x[1].intersection(candidates))).flatMap(
        get_kvp_v).reduceByKey(lambda x, y: x+y).filter(lambda x: x[1] >= int(support)).collect()
    """
    Needs rework on sorting the tuples (keys)
    """
    freq_sets = sorted(freq_sets)
    # if freq_sets is empty: break loop
    if not freq_sets:
        break
    for freq in freq_sets:
        if pass_num == 1:
            res_freq += "('"+freq[0]+"'),"
        else:
            res_freq += str(tuple(sorted(freq[0]))) + ","
    res_freq = res_freq.rstrip(',') + '\n\n'
    if pass_num == 1:
        freq_sets = set([freq[0] for freq in freq_sets])
    else:
        freq_sets = set([ff for freq in freq_sets for ff in freq[0]])
    pass_num += 1
# write file
res += "Candidates:\n" + res_can + "Frequent Itemsets:\n" + res_freq.rstrip('\n')
with open(output_path, 'w') as out_file:
    out_file.writelines(res)

"""
End timer
"""
duration = time.time() - start_time
print("Duration: "+str(duration))

