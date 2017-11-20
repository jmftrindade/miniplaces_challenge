#!/bin/sh

log=/tmp/grid_search_results.txt
tstamp=`date`
echo $tstamp > $log

# vanilla weighted majority
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt >> $log 2>&1

# using class "confidence" scores
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top1-class-scores >> $log 2>&1
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top5-class-scores >> $log 2>&1
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top1-class-scores --use-top5-class-scores >> $log 2>&1

# using class accuracies
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top1-class-accuracies >> $log 2>&1
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top5-class-accuracies >> $log 2>&1
nohup python -u weighted_majority.py -i validation/ -lf ../data/val.txt --use-top1-class-accuracies --use-top5-class-accuracies >> $log 2>&1
