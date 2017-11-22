def run():
    correct = {}
    with open('/home/labuser/miniplaces/data/train.txt') as f:
        for line in f:
            line=line.split()
            correct[line[0]] = int(line[1])

    top_1 = {}
    top_5 = {}
    top_1_total = {}
    top_5_total = {}
    for i in range(100):
        top_1[i] = 0
        top_5[i] = 0
        top_1_total[i] = 0.
        top_5_total[i] = 0.
    with open('resnet_train.txt') as f:
        for line in f:
            line = line.split()
            fname = line[0]
            guesses = [int(j) for j in line[1:]]
            true_val = correct[fname]
            top_1_total[guesses[0]] += 1
            for guess in guesses[:5]:
                top_5_total[guess] += 1
            if true_val == guesses[0]:
                top_1[true_val] += 1
            if true_val in guesses[:5]:
                top_5[true_val] += 1
    for i in range(100):
        print i, top_1_total[i], 0 if top_1_total[i] == 0 else top_1[i]/top_1_total[i], top_5_total[i], top_5[i]/top_5_total[i]

            

run()
