from pyspark import SparkContext
import os
import time
import operator
from collections import deque
from fractions import Fraction
import sys
from itertools import combinations
from copy import deepcopy

def betweenness(adjacent_graph, spark_context):
    broadcast_adjacent_graph = spark_context.broadcast(adjacent_graph)
    vertices = spark_context.parallelize(adjacent_graph.keys())
    betweenness = vertices.flatMap(lambda s: bfs(broadcast_adjacent_graph.value, s)).reduceByKey(operator.add)
    return betweenness

def bfs(graph, s):
    connected_vertices = []
    predecessors = {v:[] for v in graph}
    contribution = dict.fromkeys(graph, 0)
    level_dict = {s:0}
    contribution[s] = 1
    queue = deque([s])

    while queue:   
        v = queue.popleft()
        connected_vertices.append(v)
        curr_level = level_dict[v]
        curr_contribution = contribution[v]
        for neighbor in graph[v]:
            if neighbor not in level_dict:
                queue.append(neighbor)
                level_dict[neighbor] = curr_level + 1
            if level_dict[neighbor] == curr_level + 1:   
                contribution[neighbor] += curr_contribution
                predecessors[neighbor].append(v)

    d = dict.fromkeys(connected_vertices, 0)
    while connected_vertices:
        v = connected_vertices.pop()
        ratio = Fraction((1 + d[v]), contribution[v])
        for p in predecessors[v]:
            res = contribution[p] * ratio
            yield (p, v), res
            d[p] += res
    return

def girvan_newman(g, num_edges):
    graph = deepcopy(g)
    best_modularity = float('-inf')
    sub_graph_vertices = None
    best_sub_graph_vertices = None
    m = num_edges / 2
    while num_edges > 0:
        sub_graph_vertices = []
        vertices = list(graph.keys())
        visited = set()
        for v in vertices:
            if v in visited: continue
            sub_graph = set([v])
            visited.add(v)
            queue = deque([v])
            while queue:
                curr = queue.popleft()
                for neighbor in graph[curr]:
                    if neighbor not in sub_graph:
                        queue.append(neighbor)
                        visited.add(neighbor)
                        sub_graph.add(neighbor)
            sub_graph_vertices.append(sorted(list(sub_graph)))

        degree = {n: len(g[n]) for n in g}

        modularity = 0.0
        for sub_graph in sub_graph_vertices:
            if len(sub_graph) == 1:
                modularity -= (2 * degree[sub_graph[0]] ** 2)
            else:
                for pair in combinations(sub_graph, 2):
                    u, v = pair
                    adjacency = 1.0 if v in graph[u] else 0.0
                    modularity += (2 * (adjacency - ((degree[u] * degree[v]) / (2 * m))))

        modularity /= (2 * m)

        if modularity > best_modularity:
            best_modularity = modularity
            best_sub_graph_vertices = deepcopy(sub_graph_vertices)


        betweenness_rdd = betweenness(graph, sc)
        _, victim_betweenness = betweenness_rdd.max(lambda x: x[1])
        victim_rdd = betweenness_rdd.filter(lambda x: x[1] == victim_betweenness).map(lambda x: (tuple(sorted(x[0])), x[1])).distinct()
        for victim, _ in victim_rdd.collect():
            print(victim)
            num_edges -= 2
            graph[victim[1]].remove(victim[0])
            graph[victim[0]].remove(victim[1])
    return best_sub_graph_vertices

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel("WARN")

    threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    start = time.time()
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)
    rdd = rdd.map(lambda x: x.split(","))

    rdd = rdd.map(lambda x: (x[0], x[1]))

    user2business = rdd.groupByKey().mapValues(set)

    user_share = user2business.cartesian(user2business)

    user_share = user_share.filter(lambda x: (x[0][0] != x[1][0]) and (len(x[0][1]) + len(x[1][1]) >= threshold)).map(lambda x: ((x[0][0], x[1][0]), x[0][1] & x[1][1]))

    user_share = user_share.mapValues(len).filter(lambda x: x[1] >= threshold)

    user_edges = user_share.map(lambda x: (x[0][0], x[0][1])).cache()

    adjacent_graph = user_edges.groupByKey().mapValues(list).collectAsMap()

    num_edges = user_edges.count()

    res_rdd = betweenness(adjacent_graph, sc)

    first_betweenness_res = res_rdd.filter(lambda x: x[0][0] < x[0][1]).collectAsMap()

    res = [(k, v) for k, v in first_betweenness_res.items()]
    res.sort(key=lambda x: (-x[1], x[0]))

    with open(betweenness_output_file_path, "w") as f:
        for user, value in res:
            f.write(str(user) + "," + str(round(float(value),5)) + "\n")    
                     
    g = girvan_newman(adjacent_graph, num_edges)

    g.sort(key=lambda x: (len(x), x[0]))

    res_f = "\n".join([str(vertices)[1:-1] for vertices in g]) + "\n"
    with open(community_output_file_path,'w') as f:
        f.write(res_f)

    duration = time.time() - start
    print("Duration: ",duration)