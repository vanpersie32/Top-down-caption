"""
This script process the original image feature of tsv format into tfrecord format
Email: 18810388176@163.com
Author: CongLi
"""
import tensorflow as tf
import os
import base64
import numpy as np
import csv
import zlib
import time
from threading import thread

tf.flags.DEFINE_string('data_dir','','The directory where image features are stored')
tf.flags.DEFINE_bool('offline',True,'for offline ')
tf.flags.DEFINE_string('split_dir','','The directory where contains the split information')
tf.flags.DEFINE_integer('num_images_per_record',10000,'The number of images per tfrecord')




def main(unused_args):
    
    FLAGS = tf.flags.FLAGS
    if FLAGS.offline or len(FLAGS.split_file):
        raise ValueError('must give karpathy split information')
    split_info = None
    data_dir = FLAGS.data_dir

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    splits = ['train','val','test']
    train_data,val_data,test_data = [],[],[]

    for split in splits:
        split_file = tf.gfile.Glob(os.path.join(data_dir,'{}*'.format(split)))
        with open(os.path.join(data_dir),split_file) as f:
            reader = csv.DictReader(f,delimiter='\t',fieldnames=FIELDNAMES)
            split_data = train_data if split=='train' else (val_data if split=='val' else test_data)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])   
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                                                        dtype=np.float32).reshape((item['num_boxes'],-1))
                split_data.append(item)
    
    # read the data split information or random split 5000 images in validation set as test set
    if FLAGS.offline:
        split_dir = FLAGS.split_dir
        splits = ['train','val','test','restval']
        split_info = {}
        for split in splits:
            split_filename = os.path.join(split_dir,'coco_split.txt'.format(split))
            with open(split_filename) as f:
                filenames = [filename.strip() for filename in list(f)]
                split_info[filename] = 'val' if split=='restval' else split
    else:
        filenames = [item['file_name'] for item in val_data]
        
    
            

if __name__=='__main__':
    tf.app.run()