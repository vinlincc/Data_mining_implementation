from pyspark import SparkContext
import os
from collections import defaultdict
import time
import sys

def dfs_1(items, path, position, target, counter, sets):
    for i in range(position, len(items)):
        item = items[i]
        new_path = path + (item, )
        if len(new_path) == target:
            counter[new_path] += 1
        elif new_path in sets[len(new_path)]:
            dfs_1(items, new_path, i + 1, target, counter, sets)
    return counter

def dfs_2(items, path, position, target, counter, sets):
    for i in range(position, len(items)):
        item = items[i]
        new_path = path + (item, )
        if new_path in sets[len(new_path)]:
            counter[new_path] += 1
            if len(new_path) < target:
                dfs_2(items, new_path, i + 1, target, counter, sets)
    return counter

def my_apriori_algorithm(iterator, threshold):
    empty = set()
    empty.add(())
    sets = [empty]
    
    data = list(iterator)
    
    while len(sets[-1]) != 0:
        mapped = [dfs_1(x[1], (), 0, len(sets), defaultdict(int), sets) for x in data]
        
        flattened = [item for sublist in mapped for item in sublist.items()]
        
        reduced = defaultdict(int)
        for k, v in flattened:
            reduced[k] += v
            
        filtered = [k for k, v in reduced.items() if v >= threshold]
        
        sets.append(set(filtered))
        
    return iter([(i, x) for i, x in enumerate(sets)])

def phase2_local(iterator, sets):
    data = list(iterator)
    
    counts = defaultdict(int)
    
    for record in data:
        item_counts = dfs_2(record[1], (), 0, len(sets) - 1, counts, sets)
    
    result = list(counts.items())
    
    return iter(result)

if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    case = int(sys.argv[1])
    threshold = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    start = time.time()
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    rdd = rdd.filter(lambda x: x!= header)
    rdd = rdd.map(lambda x: x.split(','))
    if case == 2: rdd = rdd.map(lambda x: [x[1], x[0]])
    rdd = rdd.groupByKey()
    rdd = rdd.mapValues(lambda values: sorted(list(set(values))))
    n_partition = rdd.getNumPartitions()
    local_threshold = threshold // n_partition
    candidate = rdd.mapPartitions(lambda x: my_apriori_algorithm(x, local_threshold)).reduceByKey(lambda a, b: a.union(b)).collect()
    candidate.sort()
    candidate_sets = [x[1] for x in candidate]
    candidate_sets.pop()

    aggregated_results = rdd.mapPartitions(lambda x: phase2_local(x, candidate_sets)).reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] >= threshold).map(lambda x: x[0]).collect()

    aggregated_results.sort(key=lambda x: (len(x), x))

    frequent = defaultdict(list)
    for r in aggregated_results:
        frequent[len(r)].append(r)
    frequent = [(k, v) for k, v in frequent.items()]

    frequent.sort(key=lambda x: x[0])
    frequent = [x[1] for x in frequent]

    with open(output_file_path, 'w') as f:
        f.write("Candidates\n")
        for i in range(len(candidate_sets)):
            if i == 0: continue
            # if i == 1:
            #     c = [(x) for x[0] in sorted(candidate_sets[i])]
            c = str(sorted(candidate_sets[i]))[1:-1]
            if i == 1: c = c.replace(',)', ')')
            f.write("{}\n".format(c))
            f.write('\n')
        f.write('Frequent Itemsets\n')
        for i in range(len(frequent)):
            c = str(sorted(frequent[i]))[1:-1]
            if i == 0: c = c.replace(',)', ')')
            f.write("{}\n".format(c))
            f.write('\n')
    end = time.time()
    duration = end - start
    print("Duration: {}".format(duration))