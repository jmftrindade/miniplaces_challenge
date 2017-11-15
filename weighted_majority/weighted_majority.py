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


def get_network_name(filename):
    # XXX: Network name assumed to be encoded as net_file.txt
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename.split('_')[0]


def process_scores_file(filename, class_scores):
    net = get_network_name(filename)

    with open(filename) as f:
        for line in f:
            # Syntax is: <class> <#correct_guesses> <#top_1_score> ...
            tokens = line.rstrip().split(' ')
            if len(tokens) < 2:
                print 'Exiting: not enough class scores in "%s"' % (filename)
                sys.exit(-1)

            class_name = int(tokens[0])
            top_1_score = float(tokens[2])

            if net not in class_scores:
                class_scores[net] = {}
            if class_name not in class_scores[net]:
                class_scores[net][class_name] = top_1_score

    print 'scores: %s' % class_scores

    return class_scores


def process_predictions_file(predictions_filename, votes, class_scores=None):
    net = get_network_name(predictions_filename)

    if net not in WEIGHTS:
        sys.exit('Exiting: no weight configured for "%s" network.' % net)

    with open(predictions_filename) as f:
        for line in f:
            tokens = line.rstrip().split(' ')

            # Need at least the image and 1 prediction.
            if len(tokens) <= 1:
                print 'Exiting: not enough predictions in "%s".' % (
                    predictions_filename)
                sys.exit(-1)

            image = tokens[0]
            num_preds = len(tokens) - 1  # first entry is the test image
            for i in xrange(1, num_preds + 1):
                prediction = int(tokens[i])

                if image not in votes:
                    votes[image] = {}
                if prediction not in votes[image]:
                    votes[image][prediction] = 0.0

                # Linear decay.
                #decay_factor = num_preds - i
                # Exponential decay.
                decay_factor = math.exp(-i / 5)

                class_score = 1.0
                if class_scores is not None and len(class_scores) > 0:
                    class_score = class_scores[net][prediction]

                votes[image][prediction] += WEIGHTS[net] * \
                    class_score * decay_factor

    return votes


def count_votes(input_dir, labels_input_filename, use_class_scores):
    correct_answers = {}
    should_compute_acc = False
    if labels_input_filename is not None:
        should_compute_acc = True
        with open(labels_input_filename) as labels:
            for line in labels:
                line = line.rstrip().split()
                correct_answers[line[0]] = int(line[1])

    # Retrieve class scores if available.
    class_scores = {}
    if use_class_scores:
        for scores_filename in glob.glob(input_dir + '/*_scores.txt'):
            class_scores = process_scores_file(scores_filename, class_scores)

    # XXX: Filename actually matters, yuck >.<
    votes = {}
    for predictions_filename in glob.glob(input_dir + '/*_predictions.txt'):
        votes = process_predictions_file(
            predictions_filename, votes, class_scores)

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
    parser.add_argument('--use-class-scores',
                        dest='use_class_scores',
                        action='store_true')
    parser.add_argument('--no-use-class-scores',
                        dest='use_class_scores',
                        action='store_false')
    parser.set_defaults(use_class_scores=False)

    args = parser.parse_args()
    count_votes(**vars(args))
