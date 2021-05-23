# HW5 Task1
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
    for hh in hash_set:
        user_num = int(binascii.hexlify(s.encode('utf8')), 16)
        hash_val = ((hh[0] * user_num + hh[1]) % 2333333) % 69997
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
    Bloom filter
    """
    # keep track of previous users
    previous_users = set()
    # filter bit array
    A = [0] * 69997
    # get hash functions
    hash_set = random_hash(5)
    # keep track of FPR
    fpr_string = ""
    for _ in range(num_of_asks):
        # array of positive
        p = [0] * stream_size
        stream_users = bx.ask(input_path, stream_size)
        # check in the new user in already in the stream
        for i in range(len(stream_users)):
            user_num = int(binascii.hexlify(stream_users[i].encode('utf8')), 16)
            # apply boom filter
            boom = []
            for hh in hash_set:
                hash_val = ((hh[0] * user_num + hh[1]) % 2333333) % 69997
                if A[hash_val] == 0:
                    break
                else:
                    boom.append(1)
            if len(boom) == len(hash_set):
                # identify as positive
                p[i] = 1
        #calculate false positive rate
        fp = 0
        for i in range(len(stream_users)):
            if stream_users[i] not in previous_users and p[i] == 1:
                fp += 1.0
        positives = sum(p)
        fpr = fp / (len(p)-positives+fp)
        fpr_string += str(_)+","+str(fpr)+"\n"
        # construct A and update previous_users
        for user in stream_users:
            previous_users.add(user)
            for hh in hash_set:
                user_num = int(binascii.hexlify(user.encode('utf8')), 16)
                hash_val = ((hh[0] * user_num + hh[1]) % 2333333) % 69997
                A[hash_val] = 1
    res = "Time,FPR\n"
    res += fpr_string
    with open(output_path, "w") as out_file:
        out_file.writelines(res)
    """
    Stop timer
    """
    duration = time.time() - start_time
    print("Duration: " + str(duration))
