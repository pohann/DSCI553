// In my output file, scala sorting is faster than spark sorting.
// one plausible explanation is that when using spark to sort the tuples,
// it has to move data between partitions.
// But converting RDD into list and sort in scala would avoid that traffic.
// Plus the sorting method of Scala list might be optimized for different data types as well.

import org.apache.spark.{SparkContext,SparkConf}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io._
// import scala.io.Source

object task3 {
  def main(arg: Array[String]): Unit ={
    /*
    Initializing
    */
    val review_file = arg(0)
    val business_file = arg(1)
    val output_a = arg(2)
    val output_b = arg(3)

    val spark = new SparkConf().setAppName("task3").setMaster("local[*]")
    val sc = new SparkContext(spark)
    /*
    Reading file
    */
    val read_t = System.nanoTime

     val review_RDD = sc.textFile(review_file).map{ row =>
          val json_row = parse(row)
          (json_row)
     }.map(x => (pretty(render(x \"business_id")), pretty(render(x \"stars"))))
     val business_RDD = sc.textFile(business_file).map{ row =>
          val json_row = parse(row)
          (json_row)
     }.map(x => (pretty(render(x \"business_id")), pretty(render(x \"city"))))

    val read_duration = (System.nanoTime - read_t) / 1e9d
    /*
    Task3
     */
    val join = business_RDD.join(review_RDD).map(x => (x._2._1, x._2._2))
    // A find average star for each city
    val groupRDD = join.groupByKey().map(x => (x._1, x._2.toList.map(_.slice(0, 1).toInt).sum/x._2.toList.map(_.slice(0, 1).toInt).count(x => x == x))).cache()
    val res_a = groupRDD.sortBy(x => (-x._2,x._1)).collect()
    val file_a = new File(output_a)
    val bw_a = new BufferedWriter(new FileWriter(file_a))
    bw_a.write("city,stars\n")
    res_a.foreach(x => bw_a.write(x.toString.replace("\"","").replace("(","").replace(")","")+"\n"))
    bw_a.close()
    // B compare scala and spark
    val t1 = System.nanoTime
    val scala_li = groupRDD.collect().toList
    println(scala_li.sortBy(x => (-x._2, x._1)).slice(0, 10))
    val duration1 = (System.nanoTime - t1) / 1e9d + read_duration

    val t2 = System.nanoTime
    println(groupRDD.sortBy(x => (-x._2,x._1)).collect.toList.slice(0, 10))
    val duration2 = (System.nanoTime - t2) / 1e9d + read_duration
    val res_b = Map(
      "m1" -> duration1,
      "m2" -> duration2
    )
    val file_b = new File(output_b)
    val bw_b = new BufferedWriter(new FileWriter(file_b))
    bw_b.write(scala.util.parsing.json.JSONObject(res_b).toString())
    bw_b.close()
  }
}