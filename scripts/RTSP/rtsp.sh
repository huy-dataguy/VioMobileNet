#!/bin/bash


ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
-re -f concat -safe 0 -stream_loop -1 -i playlist.txt \
-vf "scale_cuda=1280:720,hwdownload,format=yuv420p" \
-c:v h264_nvenc -preset p4 -tune ll -rc cbr -b:v 1.5M -maxrate 1.5M -bufsize 3M \
-f rtsp -rtsp_transport tcp rtsp://103.78.3.29:8554/live

