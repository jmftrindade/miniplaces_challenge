def run():
    correct = {}
    with open('/home/labuser/miniplaces/data/train.txt') as f:
        for line in f:
            line=line.split()
            correct[line[0]] = int(line[1])

    top_1 = {}
    top_5 = {}
    for i in range(100):
        top_1[i] = 0
        top_5[i] = 0
    with open('resnet_train.txt') as f:
        for line in f:
            line = line.split()
            fname = line[0]
            guesses = [int(j) for j in line[1:]]
            true_val = correct[fname]
            if true_val == guesses[0]:
                top_1[true_val] += 1
            if true_val in guesses[:5]:
                top_5[true_val] += 1
    for i in range(100):
        print i, top_1[i]/1000.0, top_5[i]/1000.0

            

run()
