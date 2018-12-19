#!/usr/bin/env python3

# To combine our approach with real world images we forked the EAST boundary boxing.
# https://github.com/quasiris/EAST

import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections
import model
import io
from icdar import restore_rectangle
import lanms
from eval import resize_image, sort_poly, detect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global predictor
checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'

def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)

        args=(timer['net']*1000, timer['restore']*1000, timer['nms']*1000)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(*args))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        return ret


    return predictor


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst





if __name__ == '__main__':

    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst = get_predictor(checkpoint_path)(img)


