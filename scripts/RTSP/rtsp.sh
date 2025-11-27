#!/bin/bash
ffmpeg -re -f concat -safe 0 -stream_loop -1 -i playlist.txt \
-vf "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: text='%{localtime\:%Y-%m-%d %H\\\\\:%M\\\\\:%S}': x=10: y=10: fontsize=24: fontcolor=white: box=1: boxcolor=black@0.5" \
-c:v libx264 -preset ultrafast -tune zerolatency \
-pix_fmt yuv420p -b:v 1.5M -maxrate 1.5M -bufsize 3M \
-c:a aac -b:a 128k -ar 44100 \
-f rtsp -rtsp_transport tcp rtsp://103.78.3.29:8554/liveFight

