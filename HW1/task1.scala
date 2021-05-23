import org.apache.spark.{SparkContext,SparkConf}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io._
// import scala.io.Source

object task1 {
  def main(arg:Array[String]): Unit={
    /*
    Initializing
    */
    val input_file = arg(0)
    val output_file = arg(1)
    val spark = new SparkConf().setAppName("task1").setMaster("local[*]")
    val sc = new SparkContext(spark)
    /*
    Reading file
    */

    val RDD = sc.textFile(input_file).map{ row =>
          val json_row = parse(row)
          (json_row)
     }.cache()

    /*
    Task1
    */
    // val res = Map[String, Any]()
    // A. total number of reviews
    // res += ("n_review" -> RDD.count())
    // B. number of reviews in 2018
    // res += ("n_review_2018" -> RDD.filter(x => pretty(render(x \"date")).slice(1, 5) == "2018").count())
    // C. number of distinct users
    // res += ("n_user" -> RDD.map(x => (x \"user_id")).distinct.count())
    // D. top ten users with most reviews
    // res += ("top10_user" -> RDD.map(x => (pretty(render(x \"user_id")), 1)).reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0, 10))
    // E. number of distinct businesses
    // res += ("n_business" -> RDD.map(x => pretty(render(x \"business_id"))).distinct.count())
    // F. top ten businesses with most reviews
    // res += ("top10_business" -> RDD.map(x => (pretty(render(x \"business_id")), 1)).reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0, 10))
    // val counts = text.flatMap(line => line.split(" ")).map(word => (word,1)).reduceByKey(_+_)
    val res = Map(
      "n_review" -> RDD.count(),
      "n_review_2018" -> RDD.filter(x => pretty(render(x \"date")).slice(1, 5) == "2018").count(),
      "n_user" -> RDD.map(x => (x \"user_id")).distinct.count(),
      "top10_user" -> RDD.map(x => (pretty(render(x \"user_id")), 1)).reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0, 10),
      "n_business" -> RDD.map(x => pretty(render(x \"business_id"))).distinct.count(),
      "top10_business" -> RDD.map(x => (pretty(render(x \"business_id")), 1)).reduceByKey(_+_).sortBy(x => (-x._2, x._1)).collect.toList.slice(0,10)
    )
    // wrtie file
    val file = new File(output_file)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(scala.util.parsing.json.JSONObject(res).toString().replace("List(","[").replace("(","[").replace(")","]"))
    bw.close()
  }
}