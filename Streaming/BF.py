from blackbox import BlackBox
import binascii
import random
import math
import sys
random.seed(553)

def myhashs(s):
    result = []
    hash_function_list = []
    for _ in range(8):
        hash_function_list.append(lambda x: (random.randint(1, 69997) * x + random.randint(0, 69996)) % 69997)
    for f in hash_function_list:
        result.append(f(int(binascii.hexlify(s.encode('utf8')),16)))
    return result

if __name__ == '__main__':
    bx = BlackBox()

    input_file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    bit_array = [0] * 69997

    k = max(math.ceil((48518.22 / (stream_size * num_of_asks))), 3)

    def myhashes(s):
        result = []
        hash_function_list = []
        for _ in range(k):
            hash_function_list.append(lambda x: (random.randint(1, 69997) * x + random.randint(0, 69996)) % 69997)
        for f in hash_function_list:
            result.append(f(int(binascii.hexlify(s.encode('utf8')),16)))
        return result

    res_l = ["Time,FPR"]

    for t in range(num_of_asks):
        ground_truth = set()
        stream_users = bx.ask(input_file_name, stream_size)
        false_positive = 0
        for user in stream_users:
            res = 1 if user in ground_truth else 0
            pre = 1
            for i in myhashes(user):
                if bit_array[i] == 0:
                    pre = 0
                bit_array[i] = 1
            if pre != res: false_positive += 1
        res_l.append(str(t)+','+str(false_positive / stream_size))
    
    with open(output_file_name, 'w') as f:
        f.write('\n'.join(res_l)+'\n')