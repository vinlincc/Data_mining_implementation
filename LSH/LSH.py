
from pyspark import SparkContext
import os
import random
from itertools import combinations
import time
import operator
import sys

def minHash(x, functions):
    result = []
    for f in functions:
        result.append(min(map(f, x)))
    return result

def hashFamilies(n, num_indices):
    prime = 35053
    functions = []
    for _ in range(n):
        a = random.randint(1, num_indices)
        b = random.randint(0, num_indices)
        functions.append(lambda x, a=a, b=b, prime=prime: ((a * x + b) % prime) % num_indices)
    return functions

def candidate(x):
    x = sorted(x[1])
    res = []
    for i in combinations(x, 2):
        res.append(i)
    return res

if __name__ == '__main__':
    random.seed(42)
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    start = time.time()

    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)
    rdd = rdd.map(lambda x: x.split(","))
    rdd = rdd.map(lambda x: (x[1], x[0]))

    business_rdd = rdd.groupByKey().mapValues(list).cache()
    business_dict = business_rdd.collectAsMap()
    users = rdd.map(lambda row: row[1]).distinct().zipWithIndex()
    users_dict = users.collectAsMap()
        
    n = 60
    m = len(users_dict)

    hash_functions = hashFamilies(n, m)
    sign = business_rdd.mapValues(lambda x: [users_dict[i] for i in x]).mapValues(lambda x: minHash(x, hash_functions))

    r = 2
    b = 30
    bands = sign.flatMap(lambda x: [((i, tuple(x[1][r * i: r * (i + 1)])), x[0]) for i in range(b)])
    bands = bands.groupByKey()
    bands = bands.filter(lambda x: len(x[1]) > 1)
    bands = bands.flatMap(lambda x: candidate(x)).distinct()
    candidates = bands.collect()

    result = {}
    for bus1, bus2 in candidates:
        user1 = business_dict[bus1]
        user2 = business_dict[bus2]
        user1 = set(user1)
        user2 = set(user2)
        js = len(user1 & user2) / len(user1 | user2)
        if js >= 0.5:
            result[str(bus1) + "," + str(bus2)] = js

    result = dict(sorted(result.items(), key=operator.itemgetter(0)))
    res = "business_id_1, business_id_2, similarity\n"
    for key, values in result.items(): res += key + "," + str(values) + "\n"
    with open(output_file_path, "w") as f:
        f.writelines(res)

    end = time.time()
    print('Duration: ', end - start)