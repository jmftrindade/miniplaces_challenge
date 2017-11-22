def run():
    correct = {}
    with open('/home/labuser/miniplaces/data/val.txt') as f:
        for line in f:
            line=line.split()
            correct[line[0]] = int(line[1])

    top_1 = 0 
    top_5 = 0
    with open('resnet_val.txt') as f:
        for line in f:
            line = line.split()
            fname = line[0]
            guesses = [int(j) for j in line[1:]]
            true_val = correct[fname]
            if true_val == guesses[1]:
                top_1 += 1
            if true_val in guesses[:5]:
                top_5 += 1
    print float(top_1)/10000
    print float(top_5)/10000
            

run()
