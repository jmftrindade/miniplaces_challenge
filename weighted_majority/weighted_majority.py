import argparse
import glob
import itertools
import math
import os
import sys

# globals
WEIGHTS = {
    'resnet': 1.0,
    'inception': 1.0
}


def process_predictions_file(filename, votes):
    # network name is hardcoded in the filename:
    net = os.path.splitext(os.path.basename(filename))[0].split('_')[0]

    if net not in WEIGHTS:
        sys.exit('Exiting: no weight configured for "%s" network.' % net)

    with open(filename) as f:
        for line in f:
            tokens = line.rstrip().split(' ')

            # Need at least the image and 1 prediction.
            if len(tokens) <= 1:
                print 'Exiting: not enough predictions in "%s".' % filename
                sys.exit(-1)

            image = tokens[0]
            num_preds = len(tokens) - 1  # first entry is the test image
            for i in xrange(1, num_preds + 1):
                prediction = tokens[i]

                if image not in votes:
                    votes[image] = {}
                if prediction not in votes[image]:
                    votes[image][prediction] = 0.0

                # Prediction rank with linear decay.
                #votes[image][prediction] += WEIGHTS[net] * (num_preds - i)
                # Prediction rank with exponential decay.
                votes[image][prediction] += WEIGHTS[net] * math.exp(-i / 5)

    return votes


def count_votes(input_dir, labels_input_filename):
    votes = {}

    correct_answers = {}
    should_compute_acc = False
    if labels_input_filename is not None:
        should_compute_acc = True
        with open(labels_input_filename) as labels:
            for line in labels:
                line = line.rstrip().split()
                correct_answers[line[0]] = int(line[1])

    for filename in glob.glob(input_dir + '/*.txt'):
        votes = process_predictions_file(filename, votes)

    top_1_acc = 0
    top_5_acc = 0

    # Iterate over weighted majority votes, printing top 5 results to file.
    for image in sorted(votes.iterkeys()):
        image_votes = votes[image]
        preds = sorted(image_votes.items(), key=lambda x: (-x[1], x[0]))
#        print '%s %s' % (image, preds)

        # Output top 5 predictions.
        top_preds = []
        for i in xrange(5):
            top_preds.append(int(preds[i][0]))

        # Compute accuracy if class labels available.
        if should_compute_acc:
            print 'correct_answers[%s] = %s, top_preds = %s' % (
                image, correct_answers[image], top_preds)
            if correct_answers[image] == top_preds[0]:
                top_1_acc += 1
            if correct_answers[image] in top_preds[:5]:
                top_5_acc += 1

        print '%s %s' % (image, ' '.join(map(str, top_preds)))

    if should_compute_acc:
        print 'top-1 acc: %s' % (float(top_1_acc) / len(votes))
        print 'top-5 acc: %s' % (float(top_5_acc) / len(votes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine predictions using'
                                     ' weighted majority voting (single pass).')
    parser.add_argument('-i', '--input_dir',
                        help='Relative path of directory with input files.',
                        required=True)
    parser.add_argument('-lf', '--labels_input_filename',
                        help='Relative path of input file with class labels.',
                        required=False)

    args = parser.parse_args()
    count_votes(args.input_dir, args.labels_input_filename)
