import argparse
import glob
import itertools
import math
import os
import sys

# globals
WEIGHTS = {
    'resnet': 1.0,
    'inception': 0.5
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


def count_votes(input_dir):
    votes = {}

    for filename in glob.glob(input_dir + '/*.txt'):
        votes = process_predictions_file(filename, votes)

    # Iterate over weighted majority votes, printing top 5 results to file.
    for image in sorted(votes.iterkeys()):
        image_votes = votes[image]
        preds = sorted(image_votes.items(), key=lambda x: (-x[1], x[0]))
#        print '%s %s' % (image, preds)

        # Output top 5 predictions.
        top_preds = ''
        for i in xrange(5):
            top_preds += str(preds[i][0]) + ' '
        print '%s %s' % (image, top_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine predictions using'
                                     ' weighted majority voting (single pass).')
    parser.add_argument('-i', '--input_dir',
                        help='Relative path of directory with input files.',
                        required=True)

    args = parser.parse_args()
    count_votes(args.input_dir)
