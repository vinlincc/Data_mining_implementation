from blackbox import BlackBox
import sys
import random
random.seed(553)

if __name__ == '__main__':

    bx = BlackBox()
    input_file_name = sys.argv[1]
    size_stream = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_name = sys.argv[4]

    reservoir = []

    res_l = ["seqnum,0_id,20_id,40_id,60_id,80_id"]

    n = 0
    for t in range(num_of_asks):
        stream_users = bx.ask(input_file_name, size_stream)
        for user in stream_users:
            n += 1
            if len(reservoir) < 100:
                reservoir.append(user)
            else:
                p = random.random()
                if p < 100 / n:
                    reservoir[random.randint(0, 99)] = user
        # print(reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80])
        res_l.append(','.join([str(n), reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]]))

    with open(output_file_name, 'w') as f:
        f.write('\n'.join(res_l) + '\n')
            