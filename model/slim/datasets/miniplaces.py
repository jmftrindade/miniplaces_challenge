import tensorflow as tf
import os

slim = tf.contrib.slim
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    reader  = tf.TFRecordReader
    file_pattern = os.path.join(dataset_dir, 'miniplaces_%s_*.tfrecord' % split_name)
    keys_to_features = {
	  'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
	  'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
	  'image/class/label': tf.FixedLenFeature(
	      [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)
    num_samples = 100000 if split_name == 'train' else 10000
    print num_samples

    return slim.dataset.Dataset(data_sources=file_pattern, reader=reader, decoder=decoder, num_samples=num_samples, items_to_descriptions={'image': 'A color image of varying size.','label': 'A single integer between 0 and 4'}, num_classes=100, labels_to_names=None)


