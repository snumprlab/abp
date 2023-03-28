#!/bin/bash

# Unzip pretrained weight
cat exp/pretrained/pretrained.tar.gza* | tar xzvf -
rm exp/pretrained/pretrained.tar.gza*
mv pretrained.pth exp/pretrained

# we provide the pretrained mask rcnn
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt

curl -sc ~/cookie.txt "https://drive.google.com/uc?export=download&id=1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa" > /dev/null
curl -Lb ~/cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ~/cookie.txt`&id=1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa" -o Pretrained_Models_FILM_2.zip
unzip Pretrained_Models_FILM_2.zip
mv Pretrained_Models_FILM/maskrcnn_alfworld/objects_lr5e-3_005.pth .
mv Pretrained_Models_FILM/maskrcnn_alfworld/receps_lr5e-3_003.pth .
rm Pretrained_Models_FILM_2.zip
rm -r __MACOSX/

echo "done."
