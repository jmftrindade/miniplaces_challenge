import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from nets import nets_factory
import cv2
import numpy as np
import nets
import json
from preprocessing import preprocessing_factory
from nets import nets_factory
from google.protobuf.json_format import MessageToJson


slim = tf.contrib.slim


def inference():
    correct_answers = {}
    with open('/home/labuser/miniplaces/data/val.txt') as f:
        for line in f:
            line = line.split()
            correct_answers[line[0]] = int(line[1])
    top_1 = 0
    top_5 = 0
    with tf.Graph().as_default():
        image_bytes = tf.placeholder(tf.string, [None])
        image = tf.map_fn(lambda frame: tf.image.decode_jpeg(frame, channels=3), image_bytes, dtype=tf.uint8)
        blah = tf.map_fn(lambda frame: preprocessing_factory.get_preprocessing('inception_resnet_v2', is_training=False)(frame, 299, 299), image, dtype=tf.float32)
        net_fn = nets_factory.get_network_fn('inception_resnet_v2', 100, is_training=False)
        logits, end_points = net_fn(blah)
        #with slim.arg_scope(nets.inception_v3.inception_v3_arg_scope()):
            #logits, end_points = nets.inception_v3.inception_v3(blah, num_classes=100, is_training=False, create_aux_logits=False)
        checkpoint_path = tf.train.latest_checkpoint("/nvme/devel/tmp/inception_resnet/")
        print checkpoint_path
        with tf.Session() as sess:
            #saver = tf.train.import_meta_graph(checkpoint_path+".meta")
            saver = tf.train.Saver(slim.get_variables_to_restore())

            saver.restore(sess, checkpoint_path)
            batch_size = 250 
            im_batch = []
            im_num = []
            for i in range (1, 10001):
                im_filename = '/home/labuser/miniplaces/data/images/val/{:08d}.jpg'.format(i)
                #im = cv2.imread(im_filename)
                f = open(im_filename, 'r')
                im = f.read() 
                f.close()
                #im = cv2.resize(im, (224,224))
                im_batch.append(im)
                im_num.append(i)
                if len(im_batch) == batch_size:
                    logit = sess.run((logits), feed_dict={image_bytes:im_batch})
                    for j in range(len(im_batch)):
                        index = range(0, 100)
                        index.sort(key=lambda index: logit[j][index], reverse=True)
                        test_filename = 'val/{:08d}.jpg'.format(im_num[j])
                        if correct_answers[test_filename] == index[0]:
                            top_1 += 1
                        if correct_answers[test_filename] in index[:5]:
                            top_5 += 1
                    im_batch = []
                    im_num = []
    print float(top_1)/10000
    print float(top_5)/10000
            

            
inference()

