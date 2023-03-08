chmod +x ./building_yolo_xcode;
./building_yolo_xcode --input1="./uva_720p.mp4" --input2="./uva_720p_changed.mp4" --bmodel=./yolov5s_building_seg.bmodel --classnames=./uva.names --code_type="H264enc" --frameNum=1200 --outputName="rtmp://192.168.150.2:2335/live/test" --encodeparams="bitrate=4000" --roi_enable=1 --videoSavePath="./video_clips"
