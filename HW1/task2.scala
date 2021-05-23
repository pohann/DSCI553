import org.apache.spark.{SparkContext,SparkConf}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io._

import org.apache.spark.HashPartitioner
// import scala.io.Source

object task2 {
  def main(arg: Array[String]): Unit = {
    /*
    Initializing
    */
    val input_file = arg(0)
    val output_file = arg(1)
    val num_partition = arg(2)

    val spark = new SparkConf().setAppName("task2").setMaster("local[*]")
    val sc = new SparkContext(spark)
    /*
    Reading file
    */
     val read_t = System.nanoTime
     val RDD = sc.textFile(input_file).map{ row =>
          val json_row = parse(row)
          (json_row)
     }.cache()
     val read_duration = (System.nanoTime - read_t) / 1e9d

    /*
    Task2
     */
    val t1 = System.nanoTime
    val top10_b_old = RDD.map(x => (pretty(render(x \"business_id")), 1)).reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0,10)
    val duration1 = (System.nanoTime - t1) / 1e9d + read_duration

    val t2 = System.nanoTime
    val newRDD = RDD.map(x => (pretty(render(x \"business_id")), 1)).partitionBy(new HashPartitioner(num_partition.toInt))
    val top10_b_new = newRDD.reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0,10)
    val duration2 = (System.nanoTime - t2) / 1e9d + read_duration


    val default = Map(
        "n_partition" -> RDD.getNumPartitions,
        "n_items" -> RDD.mapPartitions(x => (Array(x.size).iterator), true).collect.toList,
        "exe_time" -> duration1
      )
    val customized = Map(
        "n_partition" -> newRDD.getNumPartitions,
        "n_items" -> newRDD.mapPartitions(x => (Array(x.size).iterator), true).collect.toList,
        "exe_time" -> duration2
      )

    val res = """{"default": """ + scala.util.parsing.json.JSONObject(default).toString().replace("List(","[").replace(")","]") +
      """, "customized": """ + scala.util.parsing.json.JSONObject(customized).toString().replace("List(","[").replace(")","]") + """}"""
    // wrtie file
    val file = new File(output_file)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res)
    bw.close()
  }
}