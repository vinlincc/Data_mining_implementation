from pyspark import SparkContext
import os
import sys
from graphframes import GraphFrame
from pyspark.sql import SparkSession
import time


if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    spark = SparkSession(sc)
    threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    start = time.time()

    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)
    rdd = rdd.map(lambda x: x.split(","))

    rdd = rdd.map(lambda x: (x[0], x[1]))

    user2business = rdd.groupByKey().mapValues(set)

    user_share = user2business.cartesian(user2business)

    user_share = user_share.filter(lambda x: (x[0][0] != x[1][0]) and (len(x[0][1]) + len(x[1][1]) >= threshold)).map(lambda x: ((x[0][0], x[1][0]), x[0][1] & x[1][1]))

    user_share = user_share.mapValues(len).filter(lambda x: x[1] >= threshold).cache()

    users = user_share.map(lambda x: x[0]).flatMap(lambda x: x).distinct().map(lambda x: (x,))

    user_edges = user_share.map(lambda x: (x[0][0], x[0][1]))

    vertices = spark.createDataFrame(users, ["id"])

    edges = spark.createDataFrame(user_edges, ["src", "dst"])

    g = GraphFrame(vertices, edges)

    result = g.labelPropagation(maxIter=5)

    r = result.rdd.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: sorted(list(x[1]))).sortBy(lambda x:(len(x), x[0]))

    a = r.collect()
    
    with open(output_file_path,'w') as f:
        for group in a:
            f.write(str(group)[1:-1] + "\n")

    duration = time.time() - start
    print("Duration: ",duration)
