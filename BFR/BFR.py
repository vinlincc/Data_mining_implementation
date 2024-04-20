from pyspark import SparkContext
import os
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import sys

def mergeRS2CS(RS_X, k_n, CS_num, CS_distribution, res_CS, cluster_increment):
    kmeans_RS = KMeans(n_clusters=5*k_n, random_state=553).fit(RS_X[:, 1:])
    tmp = defaultdict(list)
    for i, cluster in enumerate(kmeans_RS.labels_):
        tmp[cluster_increment + cluster].append(i)
    RS = {k:v for k, v in tmp.items() if len(v) == 1}
    RS_indice = [v[0] for v in RS.values()]
    CS_num_new = {k:len(v) for k, v in tmp.items() if len(v) != 1}
    res_CS_new = {}
    CS_distribution_new = {}
    for k, v in tmp.items():
        if k not in CS_num_new: continue
        d = RS_X[v, 1:]
        ids = RS_X[v, 0]
        res_CS_new[k] = list(map(int, list(ids.flatten())))
        CS_distribution_new[k] = np.stack((d.sum(axis=0), (d ** 2).sum(axis=0)))
    CS_num.update(CS_num_new)
    CS_distribution.update(CS_distribution_new)
    res_CS.update(res_CS_new)
    return RS_X[RS_indice, :]

def assign2set(point, S_num, S_distribution):
    cluster = None
    min_distance = float('inf')
    for k, v in S_distribution.items():
        centroid = v[0] / S_num[k]
        var = 0.0000000000001 + ((v[1] / S_num[k]) - centroid ** 2)
        normalize_point_square = np.divide((point - centroid) ** 2, var)
        m_distance = (normalize_point_square.sum()) ** 0.5
        if m_distance < 2 * len(point) ** 0.5:
            if m_distance < min_distance:
                cluster = k
                min_distance = m_distance
    return cluster

def mergePoint2Set(point, S_num, S_distribution, res_S):
    id = point[0]
    point = point[1:]
    c = assign2set(point, S_num, S_distribution)
    if c is not None:
        S_num[c] += 1
        S_distribution[c] += np.stack((point, (point ** 2)))
        res_S[c].append(id)
        return True
    else:
        return False
def mergeCS(CS_num, CS_distribution, res_CS):
    i = 0
    visited_cluster = set()
    while i < len(CS_num):
        tmp = [(k, v) for k, v in CS_num.items()]
        tmp.sort(key=lambda x: x[1])
        cluster, n = tmp[i]
        visited_cluster.add(cluster)
        CS_num_tmp = {k:v for k, v in CS_num.items() if k not in visited_cluster}
        CS_distribution_tmp = {k:v for k, v in CS_distribution.items() if k not in visited_cluster}
        c = assign2set(CS_distribution[cluster][0] / n, CS_num_tmp, CS_distribution_tmp)
        if c is not None:
            CS_num[c] += n
            CS_distribution[c] += CS_distribution[cluster]
            res_CS[c] += res_CS[cluster]
            del CS_num[cluster]
            del CS_distribution[cluster]
            del res_CS[cluster]
        else:
            i += 1

def mergeCS2DS(CS_num, CS_distribution, res_CS, DS_num, DS_distribution, res_DS):
    for cluster in CS_num:
        sum_CS, sumsq_CS = CS_distribution[cluster]
        n = CS_num[cluster]
        centroid_CS = sum_CS / n
        c = assign2set(centroid_CS, DS_num, DS_distribution)
        if c is not None:
            CS_num[cluster] = 0
            res_DS[c] += res_CS[cluster]
            DS_num[c] += n
            DS_distribution[c] += CS_distribution[cluster]

if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    sc = SparkContext('local[*]', 'task')
    sc.setLogLevel("WARN")
    input_file_path = sys.argv[1]
    k_n = int(sys.argv[2])
    output_file_path = sys.argv[3]
    res_f = "The intermediate results:\n"

    rdd = sc.textFile(input_file_path).cache()
    rdd = rdd.map(lambda x: x.split(",")).map(lambda x: (int(x[0]), int(x[1]), x[2:])).cache()
    label_rdd = rdd.map(lambda x: (x[0], x[1]))
    data_rdd = rdd.map(lambda x: (x[0], x[2]))
    data_rdd_list = data_rdd.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], 553)
    init_rdd = data_rdd_list[0]
    init_data_dict = init_rdd.map(lambda x: (x[0], list(map(float, x[1])))).collectAsMap()

    X = np.array([[k] + v for k, v in init_data_dict.items()])
    kmeans = KMeans(n_clusters=5*k_n, random_state=553).fit(X[:, 1:])
    tmp = defaultdict(list)
    for i, cluster in enumerate(kmeans.labels_):
        tmp[cluster].append(i)
    RS = {k:v for k, v in tmp.items() if len(v) == 1}
    RS_indice = [v[0] for v in RS.values()]
    RS_X = X[RS_indice, :]
    DS_X = np.delete(X, RS_indice, axis=0)
    kmeans_DS = KMeans(n_clusters=k_n, random_state=553).fit(DS_X[:, 1:])
    tmp = defaultdict(list)
    for i, cluster in enumerate(kmeans_DS.labels_):
        tmp[cluster].append(i)
    res_DS = defaultdict(list)
    for k, v in tmp.items():
        label = k
        row_indices = v
        for r in row_indices:
            id = int(DS_X[r][0])
            res_DS[label].append(id)
    DS_num = {k:len(v) for k, v in tmp.items()}
    DS_distribution = {}
    for k, v in tmp.items():
        d = DS_X[v, 1:]
        DS_distribution[k] = np.stack((d.sum(axis=0), (d ** 2).sum(axis=0)))
    CS_num = {}
    CS_distribution = {}
    res_CS = {}
    
    if len(RS_X) > 5*k_n:
        mergeRS2CS(RS_X, k_n, CS_num, CS_distribution, res_CS, 0)
    CS_num = {k: v for k, v in CS_num.items() if v != 0}

    res_f += "Round {4}: {0},{1},{2},{3}\n".format(int(sum(DS_num.values())), len(CS_num), int(sum(CS_num.values())), len(RS_X), 1)
    i = 2
    for data_rdd in data_rdd_list[1:]:
        data_dict = data_rdd.map(lambda x: (x[0], list(map(float, x[1])))).collectAsMap()
        X = np.array([[k] + v for k, v in data_dict.items()])
        for point in X:
            callback = mergePoint2Set(point, DS_num, DS_distribution, res_DS)
            if not callback: callback = mergePoint2Set(point, CS_num, CS_distribution, res_CS)
            if not callback: RS_X = np.append(RS_X, [point], axis=0)
        if len(RS_X) > 5*k_n:
            cluster_increment = max(CS_num.keys()) + 1 if list(CS_num.keys()) else 0
            RS_X = mergeRS2CS(RS_X, k_n, CS_num, CS_distribution, res_CS, cluster_increment)
        mergeCS(CS_num, CS_distribution, res_CS)
        CS_num = {k: v for k, v in CS_num.items() if v != 0}
        res_f += "Round {4}: {0},{1},{2},{3}\n".format(int(sum(DS_num.values())), len(CS_num), int(sum(CS_num.values())), len(RS_X), i)
        i += 1
    mergeCS2DS(CS_num, CS_distribution, res_CS, DS_num, DS_distribution, res_DS)
    CS_num = {k: v for k, v in CS_num.items() if v != 0}

    res_f += "\nThe clustering results:\n"
    res_pair = []
    for k, v in res_DS.items():
        for i in v:
            res_pair.append((int(i), k))
    for k in CS_num:
        for i in res_CS[k]:
            res_pair.append((int(i), -1))
    for r in RS_X:
        res_pair.append((int(r[0]), -1))
    res_pair.sort(key=lambda x: x[0])
    res_f += "\n".join(["{},{}".format(id, cluster) for id, cluster in res_pair])
    res_f += "\n"
    with open(output_file_path, 'w') as f:
        f.write(res_f)
