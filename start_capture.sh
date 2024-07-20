#!/bin/bash
source /home/luke/Documents/nanobox/bin/activate
cd /home/luke/Documents/capture_box
python3 capture_run.py --input_dir /media/usbdrive/motion_images --output_dir /media/usbdrive/yolo_outputs --interval 300 --confidence 0.35