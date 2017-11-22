
def read_file(filename):
    out = [] 
    with open(filename, 'r') as f:
        for line in f:
            splitline = line.split()
            out.append((splitline[0], [int(num) for num in splitline[1:]]))
    return out
correct = {}
with open('/home/labuser/miniplaces/data/val.txt') as f:
    for line in f:
        line=line.split()
        correct[line[0]] = int(line[1])

inception = read_file('inception_val.txt')
resnet = read_file('resnet_val.txt')
top_1 = 0
top_5 = 0
for i in range(len(inception)):
    inception_guess = inception[i][1][:9]
    resnet_guess = resnet[i][1][:9]
    guesses = []
    for guess in resnet_guess:
        if guess in inception_guess:
            guesses.append(guess)
    for guess in resnet_guess:
        if guess not in guesses:
            guesses.append(guess)
    if guesses[0] == correct[inception[i][0]]:
        top_1 += 1
    if correct[inception[i][0]] in guesses[:5]:
        top_5 += 1
print float(top_1)/10000
print float(top_5)/10000
