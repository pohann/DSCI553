import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.mutable.{ListBuffer, Map}
import scala.io.Source
import scala.util.Random
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import scala.collection.Set
import java.io._

import scala.collection.mutable
import scala.math.pow
import breeze.linalg._
import breeze.stats.distributions._

import scala.util.control.Breaks._

object task {
  def main(arg:Array[String]): Unit={
    /*
    Initializing
    */
    val input_path = arg(0)
    val num_cluster = arg(1).toInt
    val output_path = arg(2)

    val spark = new SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g").setAppName("task").setMaster("local[*]")
    val sc = new SparkContext(spark)
    sc.setLogLevel("ERROR")

    // Start timer
    val start_time = System.nanoTime
    /*
    BFR Algorithm
     */
    var res = "The intermediate results:\n"

    // keep track of outliers
    var RS:ListBuffer[List[Float]] = ListBuffer()
    val CS:Map[Int, Map[String, DenseVector[Float]]] = Map()
    // load data
    val raw_data:ListBuffer[List[Float]] = ListBuffer()
    for (line <- Source.fromFile(input_path).getLines()){
      raw_data += line.split(",").toList.map(x => x.toFloat)
    }
    // data: (id, vector)
    val data = for (rd <- raw_data) yield (rd(0).toInt,rd.slice(2, rd.length-1))
    // data_dict: {id:vector}
    val data_dict: Map[Int, List[Float]] = Map()
    for (d <- data){
      data_dict += (d._1 -> d._2)
    }
    // data_dict_reversed: {vector: id}
    val data_dict_reversed:Map[List[Float], Int] = Map()
    for (d <- data){
      data_dict_reversed += (d._2 -> d._1)
    }
    // way to look up dictionary
    val vector_data = for(d <- data) yield d._2
    // randomly shuffle the list to simulate random sampling
    val vector_data_new = Random.shuffle(vector_data)
    val chunk_len:Int = (vector_data_new.length/5)
    val distance_threshold:Float = pow(vector_data_new(0).length, 0.5).toFloat * 2

    // step 1: load 20% of the data randomly
    var chunk_data = vector_data_new.slice(0, chunk_len)

    // step 2: run K-means with large K
    var RDD_data:ListBuffer[Vector] = ListBuffer()
    for (dd <- chunk_data){
      RDD_data += Vectors.dense(dd.toArray.map(_.toDouble))
    }
    var RDD = sc.parallelize(RDD_data)
    var clusters = KMeans.train(RDD, num_cluster*30, 30)
    var prediction = clusters.predict(RDD).collect().toList

    // step xx: detect outliers
    var chunk_data_RDD = RDD.map(x => DenseVector(x.toArray.map(x => x.toFloat))).cache()
    var chunk_data_RDD_collect =  chunk_data_RDD.collect().toList
    var sum_sumsq = chunk_data_RDD.map(x => (x, x * x)).reduce((x,y) => (x._1+y._1, x._2+y._2))
    var centroid = sum_sumsq._1 / chunk_data.length.toFloat
    var sigma = (sum_sumsq._2 / chunk_data.length.toFloat - centroid * centroid).map(x => pow(x, 0.5).toFloat)
    val outliers:ListBuffer[Int] = ListBuffer()
    for (i <- 0 to chunk_data.length - 1){
      var z = (chunk_data_RDD_collect(i) - centroid) / sigma
      var m_distance = pow((z dot z), 0.5).toFloat
      if (m_distance > distance_threshold) {
        outliers += i
      }
    }
    println(outliers)
    // remove outliers from chunk_data and add them to RS
    for (o <- outliers){
      chunk_data -= chunk_data(o)
      RS += chunk_data(o)
    }


//    // step 3: move clusters with only one points to RS
//    // cluster_dict: {cluster id: # of points in the cluster}
//    var cluster_dict:Map[Int, Int] = Map()
//    for (pp <- prediction) {
//      if (cluster_dict.contains(pp)){
//        cluster_dict(pp) = cluster_dict(pp) + 1
//      }
//      else{
//       cluster_dict += (pp -> 1)
//      }
//    }
//    // find id of clusters that have only one point in it
//    var RS_key:ListBuffer[Int] = ListBuffer()
//    for ((k,v) <- cluster_dict) {
//      if (v < 20){
//        RS_key += k
//      }
//    }
//    // find the index of the RS points
//    var RS_index:ListBuffer[Int] = ListBuffer()
//    if (RS_key != ListBuffer()) {
//      for (key <- RS_key){
//        for (ii <- prediction.zipWithIndex.filter(_._1 == key).map(_._2)){
//          RS_index += ii
//        }
//      }
//    }
//    // append those points to RS
//    for (index <- RS_index){
//      RS += chunk_data(index)
//    }
//    // delete points from chunk_data
//    for (index <- RS_index){
//      chunk_data -= chunk_data(index)
//    }

    // step 4: run k-means with k clusters on data points not in RS
    RDD_data = ListBuffer()
    for (dd <- chunk_data){
      RDD_data += Vectors.dense(dd.toArray.map(_.toDouble))
    }
    RDD = sc.parallelize(RDD_data).cache()
    clusters = KMeans.train(RDD, num_cluster, 30)
    prediction = clusters.predict(RDD).collect().toList
    chunk_data_RDD_collect = RDD.map(x => DenseVector(x.toArray.map(x => x.toFloat))).collect().toList
    println(vector_data_new.length)
    // check the results of initial clusters
    var prediction_RDD = sc.parallelize(prediction).map(x => (x, 1)).reduceByKey((x,y) => x+y).map(x => x._2).collect().toList
    breakable{
      while (true){
        if (prediction_RDD.max.toFloat > vector_data_new.length.toFloat *(0.2) *(1.8)*(1/num_cluster.toFloat)){
          println("oh no")
          clusters = KMeans.train(RDD, num_cluster, 30)
          prediction = clusters.predict(RDD).collect().toList
          prediction_RDD = sc.parallelize(prediction).map(x => (x, 1)).reduceByKey((x,y) => x+y).map(x => x._2).collect().toList
        }
        else {
          break
        }
      }
    }

    // step 5: generate DS clusters
    // DS: {cluster id:{N:[id](float), SUM:, SUMSQ:}}
    val DS:Map[Int, Map[String, DenseVector[Float]]] = Map()
    var cluster:Int = 0
    var point:DenseVector[Float] = DenseVector(0.toFloat)
    for (i <- 0 to prediction.length - 1){
      if (DS.contains(prediction(i))){
        cluster = prediction(i)
        point = chunk_data_RDD_collect(i)
        DS(cluster)("N") = DenseVector.vertcat(DS(cluster)("N"), DenseVector(data_dict_reversed.get(chunk_data(i).toArray.toList).toList(0).toFloat))
        DS(cluster)("SUM") = DS(cluster)("SUM") + point
        DS(cluster)("SUMSQ") = DS(cluster)("SUMSQ") + point * point
      }
      else{
        DS += (prediction(i) -> Map())
        DS(prediction(i)) += ("N" -> DenseVector(data_dict_reversed.get(chunk_data(i).toArray.toList).toList(0).toFloat))
        DS(prediction(i)) += ("SUM" -> chunk_data_RDD_collect(i))
        DS(prediction(i)) += ("SUMSQ"-> chunk_data_RDD_collect(i) * chunk_data_RDD_collect(i))
      }
    }


//    // step 6: run k-means on RS to generate CS and RS
//    val CS:Map[Int, Map[String, DenseVector[Float]]] = Map()
//    if (RS != ListBuffer()) {
//      RDD_data = ListBuffer()
//      for (dd <- RS) {
//        RDD_data += Vectors.dense(dd.toArray.map(_.toDouble))
//      }
//      RDD = sc.parallelize(RDD_data).cache()
//      clusters = KMeans.train(RDD, (RS.length.toDouble*4/5).toInt + 1, 20)
//      prediction = clusters.predict(RDD).collect().toList
//      var RS_RDD = RDD.map(x => DenseVector(x.toArray.map(x => x.toFloat))).collect().toList
//      // cluster_dict: {cluster id: # of points in the cluster}
//      var cluster_dict:Map[Int, Int] = Map()
//      for (pp <- prediction) {
//        if (cluster_dict.contains(pp)){
//          cluster_dict(pp) = cluster_dict(pp) + 1
//        }
//        else{
//          cluster_dict += (pp -> 1)
//        }
//      }
//      // RS_key: index of cluster that only has one point in it
//      var RS_key:ListBuffer[Int] = ListBuffer()
//      for ((k,v) <- cluster_dict) {
//        if (v == 1){
//          RS_key += k
//        }
//      }
//      // RS_index: index of the RS points in the dataset
//      var RS_index:ListBuffer[Int] = ListBuffer()
//      if (RS_key != ListBuffer()) {
//        for (key <- RS_key){
//          RS_index += prediction.indexWhere(x => x==key)
//        }
//      }
//      // add points from RS to CS
//      for (i <- 0 to prediction.length-1){
//        if(!RS_index.contains(i)){
//          if (CS.contains(prediction(i))){
//            CS(prediction(i))("N") = DenseVector.vertcat(CS(prediction(i))("N"), DenseVector(data_dict_reversed.get(RS(i)).toList(0).toFloat))
//            CS(prediction(i))("SUM") = CS(prediction(i))("SUM") + RS_RDD(i)
//            CS(prediction(i))("SUMSQ") = CS(prediction(i))("SUMSQ") + RS_RDD(i) * RS_RDD(i)
//          }
//          else {
//            CS += (prediction(i) -> Map())
//            CS(prediction(i)) += ("N" -> DenseVector(data_dict_reversed.get(RS(i)).toList(0).toFloat))
//            CS(prediction(i)) += ("SUM" -> RS_RDD(i))
//            CS(prediction(i)) += ("SUMSQ"-> RS_RDD(i) * RS_RDD(i))
//          }
//        }
//      }
//      // update RS (remove points added to CS)
//      var new_RS: ListBuffer[List[Float]] = ListBuffer()
//      for (i <- 0 to RS.length-1) {
//        if (RS_index.contains(i)){
//          new_RS += RS(i)
//        }
//      }
//      RS = new_RS
//    }
    var round_res = round_result(DS, CS, RS, 1, vector_data_new.length)
    res += round_res
    println(res)
    for (key <- DS.keys){
      println(DS(key)("N").length)
    }

    for (r <- 2 to 5) {
      // step 7: load another 20% of data
      if (r == 5){
        chunk_data = vector_data_new.slice(chunk_len*4, vector_data_new.length)
      }
      else{
        chunk_data = vector_data_new.slice(chunk_len*(r-1), chunk_len*r)
      }

      RDD_data = ListBuffer()
      for (dd <- chunk_data) {
        RDD_data += Vectors.dense(dd.toArray.map(_.toDouble))
      }
      RDD = sc.parallelize(RDD_data)
      var chunk_data_RDD = RDD.map(x => DenseVector(x.toArray.map(x => x.toFloat))).cache()

      // step xx: detect outliers
      var chunk_data_RDD_collect =  chunk_data_RDD.collect().toList
      var sum_sumsq = chunk_data_RDD.map(x => (x, x * x)).reduce((x,y) => (x._1+y._1, x._2+y._2))
      var centroid = sum_sumsq._1 / chunk_data.length.toFloat
      var sigma = (sum_sumsq._2 / chunk_data.length.toFloat - centroid * centroid).map(x => pow(x, 0.5).toFloat)
      val outliers:ListBuffer[Int] = ListBuffer()
      for (i <- 0 to chunk_data.length - 1){
        var z = (chunk_data_RDD_collect(i) - centroid) / sigma
        var m_distance = pow((z dot z), 0.5).toFloat
        if (m_distance > distance_threshold) {
          outliers += i
        }
      }
      println(outliers)
      // remove outliers from chunk_data and add them to RS
      for (o <- outliers){
        chunk_data -= chunk_data(o)
        RS += chunk_data(o)
      }

      // step 8: assign points to DS when they're close enough to the centroid

      // keep track of points that's already assigned to DS
      var DSCS_index: ListBuffer[Int] = ListBuffer()
      RDD_data = ListBuffer()
      for (dd <- chunk_data) {
        RDD_data += Vectors.dense(dd.toArray.map(_.toDouble))
      }
      RDD = sc.parallelize(RDD_data)
      chunk_data_RDD = RDD.map(x => DenseVector(x.toArray.map(x => x.toFloat))).cache()
      chunk_data_RDD_collect =  chunk_data_RDD.collect().toList
      for (k <- 0 to chunk_data.length - 1){
        var point = chunk_data_RDD_collect(k)
        var distance_dict = m_distance_point_all(point, DS, distance_threshold)
        var m_distance = distance_dict.values.min
        var closest_cluster = 0
        for (key <- distance_dict.keys){
          if (distance_dict(key) == m_distance){
            closest_cluster = key
          }
        }

        DS(closest_cluster)("N") = DenseVector.vertcat(DS(closest_cluster)("N"), DenseVector(data_dict_reversed.get(chunk_data(k)).toList(0).toFloat))
        DS(closest_cluster)("SUM") = DS(closest_cluster)("SUM") + point
        DS(closest_cluster)("SUMSQ") = DS(closest_cluster)("SUMSQ") + point * point
        DSCS_index += k

      }


//      // distance_RDD: N, SUM, SUMSQ, index of point in the chunk_data
//      var distance_RDD = chunk_data_RDD.map(x => (DenseVector(data_dict_reversed.get(chunk_data(chunk_data_RDD_collect.indexOf(x)).toArray.toList).toList(0).toFloat), m_distance_point(x, DS, distance_threshold), x, ListBuffer(chunk_data_RDD_collect.indexOf(x)) )).filter(
//        x => x._2(1) == 1).map(x => (x._2(0), (x._1, x._3, x._3 * x._3, x._4))).reduceByKey((x, y)=> (DenseVector.vertcat(x._1,y._1), x._2+y._2,x._3+y._3, x._4 ++ y._4)).collect().toList
//      // add point to DS
//      for (dd <- distance_RDD){
//        var cluster = dd._1
//        DS(cluster)("N") = DenseVector.vertcat(DS(cluster)("N"), dd._2._1)
//        DS(cluster)("SUM") = DS(cluster)("SUM") + dd._2._2
//        DS(cluster)("SUMSQ") = DS(cluster)("SUMSQ") + dd._2._3
//        DSCS_index = DSCS_index ++ dd._2._4
//      }

      // step 9: assign remaining points to CS when they're close enough to the centroid
//      if (CS != Map() && DSCS_index.length < chunk_data.length) {
//        distance_RDD = chunk_data_RDD.map(x => (DenseVector(data_dict_reversed.get(chunk_data(chunk_data_RDD_collect.indexOf(x)).toArray.toList).toList(0).toFloat), m_distance_point(x, CS, distance_threshold), x, ListBuffer(chunk_data_RDD_collect.indexOf(x)) )).filter(x => !DSCS_index.contains(x._4(0))).filter(
//          x => x._2(1) == 1).map(x => (x._2(0), (x._1, x._3, x._3 * x._3, x._4))).reduceByKey((x, y)=> (DenseVector.vertcat(x._1,y._1), x._2+y._2,x._3+y._3, x._4 ++ y._4)).collect().toList
//        // add point to CS
//        for (dd <- distance_RDD){
//          var cluster = dd._1
//          CS(cluster)("N") = DenseVector.vertcat(CS(cluster)("N"), dd._2._1)
//          CS(cluster)("SUM") = CS(cluster)("SUM") + dd._2._2
//          CS(cluster)("SUMSQ") = CS(cluster)("SUMSQ") + dd._2._3
//          DSCS_index = DSCS_index ++ dd._2._4
//        }
//      }
      // step 10: assign points not in CS and DS to RS
      for (key <- DS.keys){
        println(DS(key)("N").length)
      }
      println("")
      var round_res = round_result(DS, CS, RS, r, vector_data_new.length)
      res += round_res
      println(res)
    }
    for (key <- DS.keys){
      println(key)
      println(DS(key)("N").length)
      println(DS(key)("SUM"))
      println(DS(key)("SUMSQ"))
    }
    val cluster_results:Map[Int,Int] = Map()
    for (cluster <- DS.keys){
      for (point <- DS(cluster)("N")){
        cluster_results += (point.toInt -> cluster)
      }
      //cluster_results += (cluster -> DS(cluster)("N").toArray.map(x => x.toInt).toSet)
    }

    res += "\nThe clustering results:\n"
    val key_set = cluster_results.keySet
    val all_index = 0 to vector_data_new.length - 1 toArray
    val index_RDD = sc.parallelize(all_index).map(x=>{
      var res:String = ""
      if (key_set contains x){
        res += x.toString + "," + cluster_results(x).toString + "\n"
      }
      else {
        res += x.toString + ",-1\n"
      }
      res
    }).reduce((x,y) => x+y)
    res += index_RDD
//    for (i <- 0 to vector_data_new.length - 1) {
//      if (key_set contains i){
//        res += i.toString + "," + cluster_results(i).toString + "\n"
//      }
//      else{
//        res += i.toString + ",-1\n"
//      }
//    }
//    for (i <- 0 to vector_data_new.length - 1) {
//      var flag = 0
//      breakable {
//        for (cluster <- cluster_results.keys) {
//          if (cluster_results(cluster) contains i.toInt) {
//            res += i.toString + "," + cluster.toString + "\n"
//            break
//          }
//          else {
//            flag += 1
//          }
//        }
//      }
//      if (flag == CS.keys.size) {
//        res += i.toString + ",-1\n"
//      }
//    }

    // write file
    val file = new File(output_path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res)
    bw.close()
    // Stop timer
    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }
  def round_result(DS:Map[Int, Map[String, DenseVector[Float]]],CS:Map[Int, Map[String, DenseVector[Float]]], RS:ListBuffer[List[Float]], Round:Int, data_size:Int): String = {
    var round_res = ""
    var DS_points = 0
    for (cluster <- DS.keys){
      DS_points += DS(cluster)("N").length
    }
    var CS_cluster = CS.keys.size
    var CS_points = 0
    for (cluster <- CS.keys){
      CS_points += CS(cluster)("N").length
    }
    var RS_points = RS.length
    if (Round == 5) {RS_points = data_size - DS_points}
    round_res += "Round " + Round.toString + ": " + DS_points.toString + "," + CS_cluster.toString + "," + CS_points.toString + "," + RS_points.toString +"\n"
    round_res
  }
  def m_distance_point(point:DenseVector[Float], sum:Map[Int,Map[String, DenseVector[Float]]], distance_threshold:Float): ListBuffer[Int] = {
    val distance_dict:Map[Int, Float] = Map()
    for (cluster <- sum.keys){
      var centroid:DenseVector[Float] = sum(cluster)("SUM") / sum(cluster)("N").length.toFloat
      var sigma:DenseVector[Float] = sum(cluster)("SUMSQ") / sum(cluster)("N").length.toFloat - centroid * centroid
      var z:DenseVector[Float] = (point - centroid) / sigma
      var m_distance = pow((z dot z), 0.5).toFloat
      distance_dict += (cluster -> m_distance)
    }
    val min_distance = distance_dict.values.min
//    println(min_distance)
    var min_distance_dict:ListBuffer[Int] = ListBuffer()
    breakable {
      for (key <- distance_dict.keys){
        if (distance_dict(key) == min_distance) {
          if (min_distance < distance_threshold) {
            min_distance_dict += key
            min_distance_dict += 1
            break
          }
          else {
            min_distance_dict += key
            min_distance_dict += -1
            break
          }
        }
      }
    }
//    println(min_distance_dict)
    min_distance_dict
  }
  def m_distance_point_all(point:DenseVector[Float], sum:Map[Int,Map[String, DenseVector[Float]]], distance_threshold:Float): Map[Int, Float] = {
    val distance_dict:Map[Int, Float] = Map()
    for (cluster <- sum.keys){
      var centroid:DenseVector[Float] = sum(cluster)("SUM") / sum(cluster)("N").length.toFloat
      var sigma:DenseVector[Float] = (sum(cluster)("SUMSQ") / sum(cluster)("N").length.toFloat - centroid * centroid).map(x => pow(x, 0.5).toFloat)
      var z:DenseVector[Float] = (point - centroid) / sigma
      var m_distance = pow((z dot z), 0.5).toFloat
      distance_dict += (cluster -> m_distance)
    }
    distance_dict
  }
}