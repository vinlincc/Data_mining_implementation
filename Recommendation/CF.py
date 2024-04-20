from pyspark import SparkContext
import os
import sys
import time
import numpy as np

if __name__ == '__main__':
    # configuration on local machine
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    # parse command line arguments
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    sc = SparkContext('local[*]', 'task2_1')
    sc.setLogLevel("ERROR")

    start = time.time()

    train_rdd = sc.textFile(train_file_path)
    header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], (x[1], float(x[2])))).cache()


    test_rdd = sc.textFile(test_file_path)
    header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).cache()

    item2user_rdd = train_rdd.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(list).cache()
    item2user_dict = item2user_rdd.collectAsMap()
    item2user_centered_rdd = item2user_rdd.mapValues(lambda x: {u: r - sum([pair[1] for pair in x]) / len(x) for u, r in x}).collectAsMap()
    user2item = train_rdd.groupByKey().mapValues(list).collectAsMap()

    def pearson_correlation(id1, id2):
        dict1 = item2user_centered_rdd.get(id1, {})
        dict2 = item2user_centered_rdd.get(id2, {})
        intersect = set(dict1.keys()) & set(dict2.keys())
        if len(intersect) < 3:
            return None
        
        nominator = sum(dict1[i] * dict2[i] for i in intersect)
        denominator = 0.00000000001 + (sum(dict1[i] ** 2 for i in intersect) ** 0.5) * (sum(dict2[i] ** 2 for i in intersect) ** 0.5)
        res = nominator / denominator
        if res < 0:
            res *= 0.1
        else:
            res *= 1.1
        return res

    def predict(uid, iid):
        if uid not in user2item: return 3.75
        
        rated_items = user2item[uid]
        
        if iid not in item2user_centered_rdd: return sum([pair[1] for pair in rated_items]) / len(rated_items)
        
        correlations = []
        ratings = []
        for iid2, r in rated_items:
            c = pearson_correlation(iid, iid2)
            if c is not None:
                correlations.append(c)
                ratings.append(r)

        if len(correlations) >= 5:
            denominator = sum(abs(s) for s, _ in zip(correlations, ratings))
            nominator = sum(s * r for s, r in zip(correlations, ratings))
            if nominator < 30:
                item_ratings = [pair[1] for pair in item2user_dict.get(iid, [])]
                return sum(item_ratings) / len(item_ratings) if item_ratings else 3.75
            else:
                return nominator / denominator
        else:
            item_ratings = [pair[1] for pair in item2user_dict.get(iid, [])]
            return sum(item_ratings) / len(item_ratings) if item_ratings else 3.75

    predictions = test_rdd.map(lambda x: (x[0], x[1], predict(x[0], x[1])))

    res = "user_id, business_id, prediction\n" + "\n".join(["{},{},{}".format(uid, iid, pred) for uid, iid, pred in predictions.collect()])

    with open(output_file_path, "w") as out_file:
        out_file.write(res)

    duration = time.time() - start
    print("Duration: "+str(duration))