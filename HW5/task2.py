# HW5 Task2
# Po-Han Chen USCID: 3988044558
# Apr 05 2021


import sys
import time
from blackbox import BlackBox
import random
import binascii


# create matrix of hash functions
def random_hash(k):
    """
    generate random hash functions
    :param k: number of hash functions
    :param n: total number of bits
    :return:
    """
    num_hash = k
    hash_set = {tt for tt in tuple(zip(random.sample(range(1, sys.maxsize-1), num_hash), random.sample(range(0, sys.maxsize-1), num_hash)))}
    return hash_set


def myhashs(s):
    result = []
    hash_set = random_hash(5)
    for hash in hash_set:
        user_num = int(binascii.hexlify(s.encode('utf8')), 16)
        hash_val = ((hash[0] * user_num + hash[1]) % 69997) % 7919
        result.append(hash_val)
    return result


if __name__ == "__main__":
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
    """
    Flajolet-Martin Algorithm
    """
    # number of hash function
    num_hash = 5
    # generate hash functions
    hash_set = list(random_hash(num_hash))
    # keep track of longest trailing zero
    zeros = [0] * num_hash
    # keep track of estimation
    compare_string = ""
    # keep track of final
    total = 0
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        current_set = set()
        for user in stream_users:
            current_set.add(user)
        # get the binary representation
        for i in range(len(stream_users)):
            user_num = int(binascii.hexlify(stream_users[i].encode('utf8')), 16)
            for j in range(len(hash_set)):
                hash_val = ((hash_set[j][0] * user_num + hash_set[j][1]) % 69997) % 7919
                hash_bin = bin(hash_val)
                trailing_zero = len(hash_bin.split("1")[-1])
                if trailing_zero > zeros[j]:
                    zeros[j] = trailing_zero
        # compute estimation
        estimate = 0
        for length in zeros:
            estimate += 2 ** length
        estimate = estimate / len(zeros)
        total += round(estimate)
        # reset zeros
        zeros = [0] * num_hash
        compare_string += str(_) + "," + str(len(current_set)) + "," + str(round(estimate)) + "\n"
    print(total/(300*num_of_asks))
    res = "Time,Ground Truth,Estimation\n"
    res += compare_string
    with open(output_path, "w") as out_file:
        out_file.writelines(res)
    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: " + str(duration))
