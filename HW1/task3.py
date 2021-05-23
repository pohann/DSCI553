# HW1 task3
# Po-Han Chen USC ID:

import json
from pyspark import SparkContext
import os
import sys
import time

# parse command line arguments
review_path = sys.argv[1]
business_path = sys.argv[2]
output_path_a = sys.argv[3]
output_path_b = sys.argv[4]

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def load_json(string):
    """
    Drop text to save space and time
    """
    json_data = json.loads(string)
    return json_data


# read part of the file
# with open(review_path) as in_file:
    # r_data = in_file.readlines(200000)
sc = SparkContext('local[*]', 'task3')
review_RDD = sc.textFile(review_path).map(load_json)
# review_RDD = sc.parallelize(r_data).map(load_json).cache()
# read part of the file
# with open(business_path) as in_file:
    # b_data = in_file.readlines(600000)
business_RDD = sc.textFile(business_path).map(load_json)
# business_RDD = sc.parallelize(b_data).map(load_json).cache()
"""
Task 3
"""
business_RDD = business_RDD.map(lambda x: (x["business_id"], x["city"])).cache()
review_RDD = review_RDD.map(lambda x: (x["business_id"], x["stars"])).cache()
join = business_RDD.join(review_RDD).map(lambda x: (x[1][0], x[1][1]))
# A find average star for each city and sort by star and city name
groupRDD = join.groupByKey().map(lambda x: (x[0], sum(x[1])/len(x[1]))).cache()
res = groupRDD.sortBy(lambda x: (-x[1], x[0])).collect()
with open(output_path_a, "w") as out_file:
    out_file.write("city,stars\n")
    res = [str(rr[0])+","+str(rr[1])+"\n" for rr in res]
    out_file.writelines(res)
# B compare Python and Spark
res = dict()
# 1. Python
start_time = time.time()
py_obj = groupRDD.collect()
sort_py = sorted(py_obj, key=lambda x: (-x[1], x[0]))
if len(sort_py) >= 10:
    print(sort_py[:10])
else:
    print(sort_py)
res["m1"] = time.time() - start_time
# 2. Spark
start_time = time.time()
print(groupRDD.sortBy(lambda x: (-x[1], x[0])).take(10))
res["m2"] = time.time() - start_time
with open(output_path_b, 'w') as out_file:
    json.dump(res, out_file)
