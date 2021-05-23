# HW1 task1
# Po-Han Chen USC ID:

import json
from pyspark import SparkContext
import os
import sys

# parse command line arguments
input_path = sys.argv[1]
output_path = sys.argv[2]

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
# with open(input_path) as in_file:
    # data = in_file.readlines(2000000)
sc = SparkContext('local[*]', 'task1')
# RDD = sc.parallelize(data).map(load_json).cache()
RDD = sc.textFile(input_path).map(load_json).cache()
"""
Task 1 
"""
res = dict()
# A. count
# print("-"*10+"count of reviews: "+str(RDD.count()))
res['n_review'] = RDD.count()
# B. number of reviews in 2018
# print(RDD.filter(lambda x: x['date'][0:4] == '2018').count())
res['n_review_2018'] = RDD.filter(lambda x: x['date'][0:4] == '2018').count()
# C. number of distinct users who wrote reviews
# print("-"*10+'distinct users: '+str(RDD.map(lambda x: x['user_id']).distinct().count()))
res['n_user'] = RDD.map(lambda x: x['user_id']).distinct().count()
# D. top 10 users with most reviews
# print("-"*10+'top 10 users: '+str(RDD.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x+y).
# sortBy(lambda x: x[1], ascending=False).take(10)))
res['top10_user'] = RDD.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x+y).\
                                  sortBy(lambda x: (-x[1], x[0])).take(10)
# E. number of distinct businesses
# print("-"*10+'distinct businesses: '+str(RDD.map(lambda x: x['business_id']).distinct().count()))
res['n_business'] = RDD.map(lambda x: x['business_id']).distinct().count()
# F. top ten businesses and their reviews
# print("-"*10+'top 10 businesses: '+str(RDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).
# sortBy(lambda x: x[1], ascending=False).take(10)))
res['top10_business'] = RDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).\
                                sortBy(lambda x: (-x[1], x[0])).take(10)
# output file
with open(output_path, 'w') as out_file:
    json.dump(res, out_file)

