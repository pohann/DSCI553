# HW5 Task3
# Po-Han Chen USCID: 3988044558
# Apr 05 2021


import sys
import time
from blackbox import BlackBox
import random

if __name__ == "__main__":
    random.seed(553)
    # parse command line arguments
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    """
    Start timer
    """
    start_time = time.time()
    bx = BlackBox()

    # keep track of the sample
    sample = []
    # keep track of the number of users arrived so far
    n = 0
    # size of the sample
    s = 100
    sequence_string = ""
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        if _ > 0:
            for user in stream_users:
                n += 1
                prob_keep = random.random()
                if prob_keep < s/n:
                    position = random.randint(0, 99)
                    sample[position] = user
        else:
            for user in stream_users:
                sample.append(user)
            n = 100
        sequence_string += str(n) + "," + sample[0] + "," + sample[20] + "," + sample[40] + "," + sample[60] + "," + sample[80] + "\n"
    res = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    res += sequence_string
    with open(output_path, "w") as out_file:
        out_file.writelines(res)
    print(res)
    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: " + str(duration))
