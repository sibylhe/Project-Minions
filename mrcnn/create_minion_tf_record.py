
# coding: utf-8

# In[1]:


import hashlib
import io
import logging
import os
import sys
import random
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

sys.path.append("/anaconda3/lib/python3.6/site-packages/tensorflow/models/research") #parent folder of object-detection

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


# #data_dir = '/Users/sibylhe/Documents/DR/image_extraction/image/maskrcnn181105/'
# #annotation_dir = data_dir + 'annotations/'
# #image_dir = data_dir + 'images/'
# #xml_dir = annotation_dir + 'xmls/'
# #mask_dir = annotation_dir + 'masks/'

# # format data paths
# import os
# import shutil
# 
# old_folders = list(set(os.listdir(data_dir)) - (set(['images', 'annotations', 'records', '.DS_Store'])))
# 
# def format_path(old_folders, prefix_to_remove):
#     with open(annotation_dir+'trainval.txt','w') as f:
#         for folder in old_folders:
#             example = folder.replace(prefix_to_remove, '', 1)
#             f.write(example+'\n')
#             xml_old_path = data_dir + folder + '/' + example +'.xml'
#             shutil.move(xml_old_path, xml_dir)
#             mask_old_dir = data_dir + folder + '/Masks/'
#             masks = os.listdir(mask_old_dir)
#             for mask in masks:
#                 if mask[-3:] == 'png':
#                     mask_old_path = mask_old_dir+mask
#                     shutil.move(mask_old_path, mask_dir)
# 
# format_path(old_folders, 'minion_')

# In[38]:


#label_map_dict = label_map_util.get_label_map_dict(annotation_dir+'minion_label_map.pbtxt')


# In[2]:


def dict_to_tf_example(data,
                       mask_dir,
                       label_map_dict,
                       image_dir,
                       mask_type='png'):
    
    img_path = image_dir + data['filename']
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['imagesize']['ncols'])
    height = int(data['imagesize']['nrows'])

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []
    masks = []
    truncated = []
    
    for obj in data['object']:
        xmin = float(obj['segm']['box']['xmin'])
        xmax = float(obj['segm']['box']['xmax'])
        ymin = float(obj['segm']['box']['ymin'])
        ymax = float(obj['segm']['box']['ymax'])
    
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
    
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
    
        if obj['occluded'] == 'yes': 
            truncated.append(int(0))
        else:
            truncated.append(int(1))
    
        mask_path = mask_dir + obj['segm']['mask']
        with tf.gfile.GFile(mask_path, 'rb') as fid:
            encoded_mask_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_mask_png)
        mask = PIL.Image.open(encoded_png_io)
        mask_np = np.asarray(mask)
        mask_remapped = (mask_np != 0).astype(np.uint8)
        masks.append(mask_remapped)
        
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
    }
    
    if mask_type == 'numerical':
        mask_stack = np.stack(masks).astype(np.float32)
        masks_flattened = np.reshape(mask_stack, [-1])
        feature_dict['image/object/mask'] = (dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
        encoded_mask_png_list = []
        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


# In[3]:


def create_tf_record(output_filename,
                     label_map_dict,
                     annotation_dir,                     
                     image_dir,
                     examples,
                     mask_type='png'):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        xml_path = annotation_dir + 'xmls/' + example + '.xml'
        mask_dir = annotation_dir + 'masks/'
        
        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        
        try: 
            tf_example = dict_to_tf_example(data,
                                            mask_dir,
                                            label_map_dict,
                                            image_dir,
                                            mask_type='png')
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)
        
        writer.close()


# In[4]:


def main(_):
    data_dir = '/Users/sibylhe/Documents/DR/image_extraction/image/maskrcnn181105/'
    image_dir = data_dir + 'images/'
    annotation_dir = data_dir + 'annotations/'
    label_map_dict = label_map_util.get_label_map_dict(annotation_dir+'minion_label_map.pbtxt')
    examples_path = annotation_dir + 'trainval.txt'
    examples_list = dataset_util.read_examples_list(examples_path)
    
    logging.info('Reading from Pet dataset.')
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.75 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    
    logging.info('%d training and %d validation examples.', len(train_examples), len(val_examples))
    logging.info('Train examples: %s', str(train_examples))
    logging.info('Validation examples: %s', str(val_examples))
    
    if not os.path.exists(data_dir + 'records'):
        os.makedirs(data_dir + 'records')
    train_output_path = data_dir + 'records/minion_train.record'
    val_output_path = data_dir + 'records/minion_val.record'
    
    create_tf_record(train_output_path,
                     label_map_dict,
                     annotation_dir,
                     image_dir,
                     train_examples,
                     mask_type='png')
    
    create_tf_record(val_output_path,
                     label_map_dict,
                     annotation_dir,
                     image_dir,
                     val_examples,
                     mask_type='png')


# In[ ]:


if __name__ == '__main__':
    tf.app.run()

