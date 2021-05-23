import java.io.{BufferedWriter, File, FileWriter}
import scala.io.Source
import scala.collection.mutable.MutableList

object task3 {
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
      r1.setSeed(553)
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
    // random object generator
    val random_object = scala.util.Random
    random_object.setSeed(553)
    // keep track of the sample
    val sample:MutableList[String] = MutableList()
    // keep track of the number of users arrived so far
    var n:Float = 0
    // size of the sample
    val s:Float = 100
    var sequence_string = ""
    for (i <- 0 to num_of_asks-1){
      val stream_users = box.ask(input_path, stream_size)
      if (i > 0){
          for (user <- stream_users){
            n += 1
            val prob_keep = random_object.nextFloat()
            if (prob_keep < s/n){
              val position = random_object.nextInt(100)
              sample(position) = user
            }
          }
      }
      else{
        for (user <- stream_users){
          sample += user
        }
        n = 100
        println(sample.length)
      }
      sequence_string += n.toInt.toString + "," + sample(0) + "," +sample(20) + "," + sample(40) + "," + sample(60) +"," + sample(80) + "\n"
    }

    var res = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    res = res + sequence_string
    // write file
    val file = new File(output_path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(res)
    bw.close()
    // Stop timer
    val duration = (System.nanoTime - start_time) / 1e9d
    println("Duration: "+duration)
  }
}
