import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io._
import java.util.Arrays
import scala.collection.immutable._
import util.control.Breaks._

object task1 {
  def main(arg:Array[String]): Unit={
    /*
    Initializing
    */
    val case_num = arg(0)
    val support = arg(1)
    val input_file = arg(2)
    val output_file = arg(3)


    val spark = new SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g").setAppName("task1").setMaster("local[*]")
    val sc = new SparkContext(spark)
    sc.setLogLevel("ERROR")
    /*
    Read file
    */
    // Start timer
    val start_time = System.nanoTime()

    // create original RDD
    val original_RDD = {
      if (case_num == "1") {
        val RDD = sc.textFile(input_file).map{
        row => (row.split(",").toList)
      }.map(x => (x(0), x(1))).filter(x => x._1 != "user_id").partitionBy(new HashPartitioner(1.toInt))
        RDD.reduceByKey(_+","+_).map(x => (x._1, x._2.split(',').toSet)).cache()
      }
      else {
        val RDD = sc.textFile(input_file).map {
          row => (row.split(",").toList.reverse)
        }.map(x => (x(0), x(1))).filter(x => x._1 != "business_id").partitionBy(new HashPartitioner(1.toInt))
        RDD.reduceByKey(_ + "," + _).map(x => (x._1, x._2.split(',').toSet)).cache()
      }
    }
    val support_ratio = support.toFloat / original_RDD.count()
    // a-priori
    var res_can = "Candidates:\n"
    var res_freq = "Frequent Itemsets:\n"
    var pass_num = 1
    var freq_sets_flat: Set[String] = List("1").toSet
    var freq_sets_2: Set[List[String]] = Set(List("1"))
    var freq_sets : Set[Any] = Set(List("1"))
    var can_sets : Set[Any] = Set(List("1"))
    val basket_RDD = original_RDD

    breakable {
      while (true) {
        if (pass_num == 1){
          // find candidates
          val candidates = basket_RDD.mapPartitions(iterator => {
            val iter_list = iterator.toList
            val size = iter_list.size
            val res = iter_list.map(x => (x._2, size))
            res.iterator
          }).filter(x => x._1 != Set()).flatMap(x => {
            x._1.toList zip List.fill(x._1.size)(List(1 / x._2.toFloat)).flatten
          }).mapPartitions(iterator => {
            val iter_list = iterator.toList
            val dict = scala.collection.mutable.Map[Any, Float]()
            for (x <- iter_list) {
              dict(x._1) = (dict getOrElse(x._1, 0.toFloat)) + x._2
            }
            dict.toList.iterator
          }).filter(x => x._2 >= support_ratio).map(x => x._1.toString).collect().toSet.toList.sortWith(_<_)
          if (candidates == List()) break
          can_sets = candidates.toSet
          res_can += candidates.toString.replace("List","").replace(
            ", ","','").replace("(","('").replace(
            ")","')").replace("','","'),('") + "\n\n"

          val freq_items = basket_RDD.map(x => (x._1, x._2.intersect(can_sets.map(_.toString)))).flatMap(x => {
            x._2.toList zip List.fill(x._2.size)(List(1)).flatten
          }).reduceByKey(_ + _).filter(x => x._2 >= support.toFloat).map(x => x._1).collect().toSet.toList.sortWith(_<_)

          if (freq_items == List()) break
          res_freq += freq_items.toString.replace("List","").replace(
            ", ","','").replace("(","('").replace(
            ")","')").replace("','","'),('") + "\n\n"
          freq_sets = freq_items.toSet

        }

        if (pass_num == 2){
          val basket_RDD = original_RDD.map(x => (x._1, x._2.intersect(freq_sets.map(_.toString)))).map(
            x => (x._1, x._2.subsets(pass_num).map(_.toList.sortWith(_<_)).toSet))
          // find candidates
          val candidates = basket_RDD.mapPartitions(iterator => {
            val iter_list = iterator.toList
            val size = iter_list.size
            val res = iter_list.map(x => (x._2, size))
            res.iterator
          }).filter(x => x._1 != Set()).flatMap(x => {
            x._1.toList zip List.fill(x._1.size)(List(1 / x._2.toFloat)).flatten
          }).mapPartitions(iterator => {
            val iter_list = iterator.toList
            val dict = scala.collection.mutable.Map[List[String], Float]()
            for (x <- iter_list) {
              dict(x._1) = (dict getOrElse(x._1, 0.toFloat)) + x._2
            }
            dict.toList.iterator
          }).filter(x => x._2 >= support_ratio).map(x => x._1).collect().toSet.toList.sortWith(_.reduce(_+_) < _.reduce(_+_))
          if (candidates == List()) break
          res_can += candidates.toString.replace("List(List","").replace(
            "))",")").replace(" List","").replace(
            "(","('").replace(")","')").replace(
            ", ","', '") + "\n\n"
          var can_sets_2: Set[List[String]] = candidates.toSet


          val freq_items = basket_RDD.map(x => (x._1, x._2.intersect(can_sets_2))).flatMap(x => {
            x._2.toList zip List.fill(x._2.size)(List(1)).flatten
          }).reduceByKey(_ + _).filter(x => x._2 >= support.toFloat).map(x => x._1).collect().toSet.toList.sortWith(_.reduce(_+_) < _.reduce(_+_))
          if(freq_items == List()) break
          res_freq += freq_items.toString.replace("List(List","").replace(
            "))",")").replace(" List","").replace(
            "(","('").replace(")","')").replace(
            ", ","', '") + "\n\n"
          freq_sets_2 = freq_items.toSet
          freq_sets_flat = for (ff <- freq_sets_2; fff <- ff) yield fff
          // println(freq_sets_flat)
        }

        if (pass_num >= 3) {
          val basket_RDD = original_RDD.map(x => (x._1, x._2.intersect(freq_sets_flat))).map(
            x => (x._1, x._2.subsets(pass_num - 1).map(_.toList.sortWith(_<_)).toSet)).map(
            x => (x._1, x._2.intersect(freq_sets_2))).map(
          x => (x._1, for (xx <- x._2 ; xxx <- xx) yield xxx)).map(
          x => (x._1, x._2.subsets(pass_num).map(_.toList.sortWith(_<_)).toSet))

          val candidates = basket_RDD.mapPartitions(iterator => {
            val iter_list = iterator.toList
            val size = iter_list.size
            val res = iter_list.map(x => (x._2, size))
            res.iterator
          }).filter(x => x._1 != Set()).flatMap(x => {
            x._1.toList zip List.fill(x._1.size)(List(1 / x._2.toFloat)).flatten
          }).mapPartitions(iterator => {
            val iter_list = iterator.toList
            val dict = scala.collection.mutable.Map[List[String], Float]()
            for (x <- iter_list) {
              dict(x._1) = (dict getOrElse(x._1, 0.toFloat)) + x._2
            }
            dict.toList.iterator
          }).filter(x => x._2 >= support_ratio).map(x => x._1).collect().toSet.toList.sortWith(_.reduce(_+_) < _.reduce(_+_))
          if (candidates == List()) break
          res_can += candidates.toString.replace("List(List","").replace(
            "))",")").replace(" List","").replace(
            "(","('").replace(")","')").replace(
            ", ","', '") + "\n\n"
          var can_sets_2: Set[List[String]] = candidates.toSet


          val freq_items = basket_RDD.map(x => (x._1, x._2.intersect(can_sets_2))).flatMap(x => {
            x._2.toList zip List.fill(x._2.size)(List(1)).flatten
          }).reduceByKey(_ + _).filter(x => x._2 >= support.toFloat).map(x => x._1).collect().toSet.toList.sortWith(_.reduce(_+_) < _.reduce(_+_))
          if(freq_items == List()) break
          res_freq += freq_items.toString.replace("List(List","").replace(
            "))",")").replace(" List","").replace(
            "(","('").replace(")","')").replace(
            ", ","', '") + "\n\n"
          freq_sets_2 = freq_items.toSet
          freq_sets_flat = for (ff <- freq_sets_2; fff <- ff) yield fff

        }
        pass_num += 1
      }
    }

    val file = new File(output_file)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res_can+res_freq)
    bw.close()
    //print total time
    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }
}