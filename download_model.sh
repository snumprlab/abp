#!/bin/bash

# Unzip pretrained weight
cat exp/pretrained/pretrained.tar.gza* | tar xzvf -
rm exp/pretrained/pretrained.tar.gza*
mv pretrained.pth exp/pretrained

echo "done."
