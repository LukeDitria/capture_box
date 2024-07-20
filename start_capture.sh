#!/bin/bash
# Make this file runnable chmod +x start_capture.sh
#Needed to setup automatic mounting of usb in fstab
source /home/luke/Documents/nanobox/bin/activate
cd /home/luke/Documents/capture_box
python capture_run.py --input_dir /media/usb/motion_images --output_dir /media/usb/yolo_outputs --interval 300 --confidence 0.5