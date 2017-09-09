from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=("yolo_output_label",), context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))],
            label_shapes=[('yolo_output_label', (1, 2, 5))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        # import time
        # time.sleep(5) # delays for 5 seconds
        # print("Stop sleep")
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        # tmp_result = []
        # result = []
        # start = timer()
        # for det in self.mod.iter_predict(det_iter):
        #     det = det[0][0][0].as_in_context(mx.cpu())
        #     tmp_result.append(det)
        # time_elapsed = timer() - start
        # if show_timer:
        #     print("Detection time for {} images: {:.4f} sec".format(
        #         num_images, time_elapsed))
        # for det in tmp_result:
        #     det = det.asnumpy()
        #     res = det[np.where(det[:, 0] >= 0)[0]]
        #     result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()

    def gen_coco_json(self, im_list, thresh, save_json):
        import cv2
        import json
        import os
        dets = self.im_detect(im_list)
        print('detect ok')
        key_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
                    75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        key_map = {}
        for i in xrange(len(key_list)):
            key_map[i] = key_list[i]
        results = []
        for k, det in enumerate(dets):
            img_name = im_list[k]
            img = cv2.imread(im_list[k])
            height = img.shape[0]
            width = img.shape[1]
            for i in range(det.shape[0]):
                cls_id = int(det[i,0])
                cls_id = key_map[cls_id]
                if cls_id >= 0:
                    score = det[i,1]
                    if score > thresh:
                        xmin = int(det[i, 2] * width)
                        ymin = int(det[i, 3] * height)
                        xmax = int(det[i, 4] * width)
                        ymax = int(det[i, 5] * height)
                        h = ymax - ymin
                        w = xmax - xmin

                        result = {}
                        result['image_id'] = os.path.splitext(os.path.basename(img_name))[0]
                        result['category_id'] = cls_id
                        rect_value = [xmin, ymin, w, h]
                        result['bbox'] = rect_value
                        result['score'] = float(score)

                        results.append(result)
                    else:
                        print("score %.4f" % score)
                else:
                    print("clssid %d" % cls_id)

        with open('result.json', 'w') as fp:
            json.dump(results, fp, indent=4)
        #print(json.dumps(results, indent=4))
    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)
