import scala.io.Source
import java.lang.Math.abs
import scala.collection.mutable.MutableList
import java.io._


object task2 {
  def main(arg: Array[String]): Unit = {
    /*
    Initializing
    */
    val input_path = arg(0)
    val stream_size = arg(1).toInt
    val num_of_asks: Int = arg(2).toInt
    val output_path = arg(3)

    class Blackbox {
      private val r1 = scala.util.Random

      def ask(filename: String, num: Int): Array[String] = {
        val input_file_path = filename

        val lines = Source.fromFile(input_file_path).getLines().toArray
        var stream = new Array[String](num)

        for (i <- 0 to num - 1) {
          stream(i) = lines(r1.nextInt(lines.length))
        }
        return stream
      }
    }
    /*
    Flajolet-Martin Algorithm
     */
    // Start timer
    val start_time = System.nanoTime
    // initialize Blackbox
    val box = new Blackbox
    // number of hash function
    val num_hash = 5
    // get hash function
    val hash_set = random_hash(num_hash)
    // keep track of longest trailing zero
    var zeros = MutableList.fill(num_hash)(0)
    // keep track of estimation
    var compare_string = ""
    // keep track of final
    var total:Double = 0
    for (i <- 0 to num_of_asks-1) {
      val stream_users = box.ask(input_path, stream_size)
      var current_set: Set[String] = Set()
      for (user <- stream_users) {
        current_set += user
      }
      // get the binary representation
      for (j <- 0 to stream_users.length-1){
        val user_num = abs(stream_users(j).hashCode) % 119267
        for (k <- 0 to hash_set.length-1){
          val hash_val = ((hash_set(k)(0) * user_num + hash_set(k)(1)) % 69997) % 3881
          val hash_bin = hash_val.toBinaryString
          var trailing_zero = 0
          if (hash_bin.last == "1" || !hash_bin.contains("0")){
            trailing_zero = 0
          }
          else {
            trailing_zero = hash_bin.split("1").last.length
          }
          if (trailing_zero > zeros(k)) {
            zeros(k) = trailing_zero
          }
        }
      }
      // compute estimation
      var estimate:Double = 0.0
      for (length <- zeros) {
        estimate += scala.math.pow(2, length)
      }
      estimate = estimate / zeros.length
      total += estimate
      // reset zeros
      zeros = MutableList.fill(num_hash)(0)

      compare_string = compare_string + i.toString + "," + current_set.size.toString + "," + estimate.round.toString + "\n"
    }
    println(total/(300*num_of_asks))
    var res = "Time,Ground Truth,Estimation\n"
    res = res + compare_string
    // write file
    val file = new File(output_path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res)
    bw.close()
    // Stop timer
    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }
  def random_hash(number_of_hash: Int): MutableList[List[Int]] = {
    val r = scala.util.Random
    val hash_set:MutableList[List[Int]]  = MutableList()
    var a = 0
    var b = 0
    for (i <- 0 to number_of_hash-1) {
      a = abs(r.nextInt(1000))
      b = abs(r.nextInt(1000))
      hash_set += List(a, b)
    }
    hash_set
  }
  def myhashs(user_string: String): Array[Int] = {
    var result:Array[Int] = Array()
    val hash_set = random_hash(5)
    val user_num = abs(user_string.hashCode) % 119267
    for (hash <- hash_set){
      val hash_val = ((hash(0) * user_num + hash(1)) % 69997) % 3881
      result = result :+ hash_val
    }
    result
  }
}