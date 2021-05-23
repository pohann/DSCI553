import org.apache.spark.sql.SparkSession
import java.io._
import scala.collection.immutable._
import org.graphframes.GraphFrame

object task1 {
  def main(arg:Array[String]): Unit={
    /*
    Initializing
    */
    val threshold = arg(0).toInt
    val input_path = arg(1)
    val output_path = arg(2)

    /*
    Create SparkContext
     */
    val spark:SparkSession = SparkSession.builder().master("local[*]").appName("task1").getOrCreate()
    // val spark = new SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g").setAppName("task1").setMaster("local[*]")
    // val sc = new SparkContext(spark)
    spark.sparkContext.setLogLevel("error")
    // Start timer
    val start_time = System.nanoTime
    /*
    Read data
     */
    // ub_RDD: [uid, bid]
    val ub_RDD = spark.sparkContext.textFile(input_path).map(row => (row.split(",").toList)).map(
      x => (x(0),x(1))).filter(x => x._1 != "user_id")
    // uid_dict: {uid:num, ... }
    val uid_set = ub_RDD.map(x => x._1).distinct().collect().toList
    val uid_range = (0 until uid_set.length).toList
    val uid_dict = (uid_set zip uid_range).toMap
    val uid_dict_reversed = (uid_range zip uid_set).toMap
    // bid_dict: {bid:[num], ... }
    val bid_set = ub_RDD.map(x => x._2).distinct().collect().toList
    val bid_range = (0 until bid_set.length).toList
    val bid_dict = (bid_set zip bid_range).toMap
    // bu_RDD: (bid, uid)
    val bu_RDD = ub_RDD.map(x => (bid_dict(x._2), uid_dict(x._1)))
    // bu_joined: (uid1,uid2) pairs of uid that has more than 7 co-rated items
    // assume there's no duplicate in the original file of uid, bid pairs
    // remember to filter out pairs where uid1 == uid2
    // x[1]/2 to get the real count of co-rated bid
    val bu_joined = bu_RDD.join(bu_RDD).map(x => x._2).filter(x => x._1 != x._2).map(
      x => x.toString.replace("(","").replace(")","").split(",").toList).map(
      x => x.map(x => x.toInt)).map(x => (x.sorted, 1)).reduceByKey(_ + _).map(x => (x._1, x._2/2)).filter(
      x => x._2 >= threshold).map(x => x._1)
    val bu_node = bu_joined.flatMap(x => List(x, List(x(1), x(0)))).map(x => (x(0), 0)).reduceByKey(_ + _).map(x => x._1).collect().toList
    // create graph frame
    // vertices: collection of uid
    // edges: pair of uid that are connected
    val ver_columns = Seq("id", "redundant")
    val vertices_old = spark.createDataFrame(bu_node.map(x => (x, 0))).toDF(ver_columns:_*)
    val vertices = vertices_old.drop("redundant")
    val edg_columns = Seq("src", "dst")
    val edges = spark.createDataFrame(bu_joined.map(x => (x(0), x(1)))).toDF(edg_columns:_*)
    val g = GraphFrame(vertices, edges)
    // detect community
    // result: label (community), [uid1,...] -> [uid1, ...]
    val result = g.labelPropagation.maxIter(5).run()
    val community = result.rdd.map(x => (x(1), List(x(0)))).reduceByKey((x, y) => List.concat(x, y)).map(
      x => x._2).collect().toList.map(x => x.map(xx => uid_dict_reversed(xx.toString.toInt)))

    val res_list = community.map(x => x.sorted).sortBy(x => (x.length, x(0)))
    var res_string = ""
    for (c <- res_list) {
      for (cc <- c) {
        res_string = res_string + "'" + cc + "', "
      }
      res_string = res_string.stripSuffix(", ") + "\n"
    }
    // write file
    val file = new File(output_path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res_string)
    bw.close()
    // Stop timer
    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }
}