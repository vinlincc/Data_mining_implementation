from blackbox import BlackBox
import binascii
import random
import sys
import time
random.seed(553)


def myhashs(s):
    k = 380
    hash_function_list = []
    for i in range(k):
        hash_function_list.append(lambda x: (random.randint(1, 997) * x + random.randint(0, 996)) % 997)
    result = []
    for f in hash_function_list:
        result.append(f(int(binascii.hexlify(s.encode('utf8')),16)))
    return result

def trailing(v):
    return (v & -v).bit_length() - 1

def FM_update(hash_res, res):
    for i in range(len(res)):
        res[i] = max(res[i], 2 ** trailing(hash_res[i]))
    return res

if __name__ == '__main__':
    start = time.time()
    bx = BlackBox()

    input_file_name = sys.argv[1]
    size_stream = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    res_l = ['Time,Ground Truth,Estimation']

    prediction = 0
    for t in range(num_of_asks):
        ground_truth = set()
        stream_users = bx.ask(input_file_name, size_stream)
        res = [1] * 380
        for user in stream_users:
            ground_truth.add(user)
            hash_res = myhashs(user)
            res = FM_update(hash_res, res)
        chunk_avg = []
        for i in range(19):
            chunk_avg.append(sum(res[i*20:(i+1)*20]) // 20)
        chunk_avg.sort()
        prediction = chunk_avg[9]
        # print(len(ground_truth), prediction)
        res_l.append(str(t)+','+str(len(ground_truth))+','+str(prediction))

    with open(output_file_name, 'w') as f:
        f.write('\n'.join(res_l)+'\n')
    #print(str(time.time() - start))
