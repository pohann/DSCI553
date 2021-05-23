import scala.io.Source
import java.lang.Math.abs
import scala.collection.mutable.MutableList
import scala.util.control.Breaks._
import java.io._


object task1 {
  def main(arg:Array[String]): Unit={
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

    // Start timer
    val start_time = System.nanoTime
    // initialize Blackbox
    val box = new Blackbox
    /*
    Bloom filter
     */
    // keep track of previous users
    var previous_users: Set[String] = Set()
    // filter bit array
    val A = MutableList.fill(69997)(0)
    // get hash function
    val hash_set = random_hash(2)
    // keep track of FPR
    var fpr_string = ""
    for (i:Int <- 0 to num_of_asks-1){
      val stream_users = box.ask(input_path, stream_size)
      // array of positive
      val p = MutableList.fill(stream_size)(0)
      for (j:Int <- 0 to stream_users.length-1){
        val user_num = abs(stream_users(j).hashCode) % 119267
        val boom: MutableList[Int] = MutableList()
        breakable {
          for (hash <- hash_set) {
            val hash_val = (hash(0) * user_num + hash(1)) % 69997
            if (A(hash_val) == 0) {
              break
            }
            else {
              boom += 1
            }
          }
        }
        if (boom.length == hash_set.length){
          // identify as positive
          p(j) = 1
        }
      }
      // calculate false positive rate
      var fp = 0.toFloat
      for (k <- 0 to stream_users.length-1){
        if (!previous_users.contains(stream_users(k)) &&  p(k) == 1) {
          fp += 1
        }
      }
      val positives = p.sum.toFloat
      val fpr = fp/(p.length-positives+fp)
      // construct A and update previous_users
      for (l <- 0 to stream_users.length - 1) {
        previous_users += stream_users(l)
        for (hash <- hash_set) {
          val user_num = abs(stream_users(l).hashCode) % 119267
          val hash_val = (hash(0) * user_num + hash(1)) % 69997
          A(hash_val) = 1
        }
      }
      fpr_string = fpr_string + i.toString + "," + fpr.toString + "\n"
    }
    var res:String = "Time,FPR\n"
    res = res + fpr_string
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
    val hash_set = random_hash(2)
    val user_num = abs(user_string.hashCode) % 119267
    for (hash <- hash_set){
      val hash_val = (hash(0) * user_num + hash(1)) % 69997
      result = result :+ hash_val
    }
    result
  }
}

