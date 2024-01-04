import argparse


def parse_cmd_arg():
    parser = argparse.ArgumentParser(description='Parser to read your config file')
    # parser.add_argument('--config', type=str, default='./config/instance_segmentation/mask_rcnn_R_50_FPN_3x_handwritten.yaml', help='path to config file')
    parser.add_argument('--config', type=str, default='./config/instance_segmentation/mask_rcnn_R_50_FPN_3x_kaiti.yaml', help='path to config file')
    
    return parser.parse_args()
