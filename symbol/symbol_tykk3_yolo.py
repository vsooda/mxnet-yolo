"""
Reference:
Redmon, Joseph, and Ali Farhadi. "YOLO9000: Better, Faster, Stronger."
"https://arxiv.org/pdf/1612.08242.pdf"
"""
import mxnet as mx
from symbol_tykk3 import get_symbol as get_tykk3
from symbol_tykk3 import conv_act_layer

def get_symbol(num_classes=20, nms_thresh=0.5, force_nms=False, **kwargs):
    bone = get_tykk3(num_classes=num_classes, **kwargs)
    conv6_5 = bone.get_internals()["conv6_5_output"]
    # anchors
    #anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    anchors = [0.738768, 0.874946, 2.42204, 2.65704, 4.30971, 7.04493, 10.246, 4.59428, 12.6868, 11.8741]
    num_anchor = len(anchors) // 2

    # extra layers
    conv7_1 = conv_act_layer(conv6_5, 'conv7_1', 512, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')

    pred = mx.symbol.Convolution(data=conv7_1, name='conv_pred', kernel=(1, 1),
        num_filter=num_anchor * (num_classes + 4 + 1))

    out = mx.contrib.symbol.YoloOutput(data=pred, num_class=num_classes,
        num_anchor=num_anchor, object_grad_scale=5.0, background_grad_scale=1.0,
        coord_grad_scale=1.0, class_grad_scale=1.0, anchors=anchors,
        nms_topk=400, warmup_samples=12800, name='yolo_output')
    return out
