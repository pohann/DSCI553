# HW1 task2
# Po-Han Chen USC ID:

import json
from pyspark import SparkContext
import os
import sys
import time

# parse command line arguments
input_path = sys.argv[1]
output_path = sys.argv[2]
num_partition = int(sys.argv[3])

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def load_json(string):
    """
    Drop text to save space and time
    """
    json_data = json.loads(string)
    json_data.pop("text", None)
    return json_data


# read part of the file
# with open(input_path) as in_file:
    # data = in_file.readlines(300000)
sc = SparkContext('local[*]', 'task2')
RDD = sc.textFile(input_path).map(lambda x: load_json(x)).cache()
# RDD = sc.parallelize(data).map(load_json).cache()
"""
Task 2
"""
res = dict()
# using old method
res["default"] = dict()
res["default"]["n_partition"] = RDD.getNumPartitions()
res["default"]["n_items"] = RDD.mapPartitions(lambda iterator: [len(list(iterator))], True).collect()
start_time = time.time()
top10_business = RDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).\
                                sortBy(lambda x: (-x[1], x[0])).take(10)
res["default"]["exe_time"] = time.time() - start_time
print("---- %s seconds ----" % (time.time() - start_time))
# using new method: partition by key
res["customized"] = dict()
start_time = time.time()
newRDD = RDD.map(lambda x: (x['business_id'], 1)).partitionBy(num_partition, lambda x: ord(x[0][0])).cache()
top10_business = newRDD.reduceByKey(lambda x, y: x+y).sortBy(lambda x: (-x[1], x[0])).take(10)
res["customized"]["exe_time"] = time.time() - start_time
print("---- %s seconds ----" % (time.time() - start_time))
res["customized"]["n_partition"] = newRDD.getNumPartitions()
res["customized"]["n_items"] = newRDD.mapPartitions(lambda iterator: [len(list(iterator))], True).collect()
# output file
with open(output_path, 'w') as out_file:
    json.dump(res, out_file)

