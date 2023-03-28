#!/bin/bash

# Unzip pretrained weight
cat exp/pretrained/pretrained.tar.gza* | tar xzvf -
rm exp/pretrained/pretrained.tar.gza*
mv pretrained.pth exp/pretrained

# we provide the pretrained mask rcnn
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
echo "done."
