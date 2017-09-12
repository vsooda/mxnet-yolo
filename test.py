import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
from detect.detector import Detector

#CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush')

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=False):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    if net is not None:
        prefix = prefix + "_" + net.strip('_yolo') + '_' + str(data_shape)
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(num_classes=len(CLASSES), nms_thresh=nms_thresh, force_nms=force_nms, nms_topk=-1)
    detector = Detector(net, prefix, epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='darknet19_yolo',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'yolo2'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=416,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=0,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=0,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=0,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.005,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.4,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    args = parser.parse_args()
    return args

def get_file_list(path):
    path = os.path.abspath(path)
    files = os.listdir(path)
    for index, item in enumerate(files):
        files[index] = os.path.join(path, files[index])
    files = sorted(files)
    return files

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    #test_dir = 'data/demo/'
    #test_dir = '/home/sooda/data/COCO/images/test2015/'
    #image_list = get_file_list(test_dir)
    test_files_test = "/home/sooda/data/COCO/test-dev.txt"

    #ftest = open(test_files_test, 'r')
    #image_list = ftest.readlines()
    image_list = []
    count = 0
    with open(test_files_test) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            image_list.append(line)
            #count = count + 1
            #if count > 10:
            #    break

    for img in image_list:
        print img

    assert len(image_list) > 0, "No valid image specified to detect"

    network = None if args.deploy_net else args.network
    detector = get_detector(network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, args.nms_thresh, args.force_nms)

    # run detection
    #detector.detect_and_visualize(image_list, args.dir, args.extension,CLASSES, args.thresh, args.show_timer)
    detector.gen_coco_json(image_list, args.thresh, 'coco_results.json')
