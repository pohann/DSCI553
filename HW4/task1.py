# HW4 Task1
# Po-Han Chen USCID: 3988044558
# Mar 25 2021


from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import os
import sys
import time

from graphframes import *

from functools import reduce

# configuration on local machine
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

if __name__ == "__main__":
    # parse command line arguments
    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    """
    Start timer
    """
    start_time = time.time()
    conf = SparkConf().setMaster("local").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

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
    bu_joined = bu_RDD.join(bu_RDD).map(lambda x: x[1]).filter(lambda x: x[0] != x[1]).map(
        lambda x: (tuple(sorted(x)), 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], x[1]/2)).filter(
        lambda x: x[1] >= threshold).map(lambda x: x[0])

    bu_single = bu_joined.flatMap(lambda x: [x, (x[1], x[0])]).map(lambda x: (x[0], set([x[1]]))).reduceByKey(
        lambda x, y: 0).map(lambda x: x[0])
    # create graph frame
    # vertices: collection of uid
    # edges: pair of uid that are connected
    vertices = sqlContext.createDataFrame(map(lambda x: tuple([x]), bu_single.collect()), ["id"])
    edges = sqlContext.createDataFrame(bu_joined, ["src", "dst"])
    g = GraphFrame(vertices, edges)
    # g.inDegrees.show()
    # detect community
    # result: label (community), [uid1,...] -> [uid1, ...]
    result = g.labelPropagation(maxIter=5)
    result = result.rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x+y).map(
        lambda x: (x[0], ["'"+uid_dict_reversed[xx]+"'" for xx in x[1]])).map(lambda x: sorted(x[1])).collect()
    result.sort(key=lambda x: (len(x), x))

    # write result
    res = ""
    for rr in result:
        rr_str = reduce(lambda x, y: x+", "+y, rr)
        res += rr_str + "\n"
    with open(output_path, "w") as out_file:
        out_file.write(res)

    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: "+str(duration))
