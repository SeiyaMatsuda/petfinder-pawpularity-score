#!/bin/bash

#swintransformer 224
cd swin224
python swin224.py
cd ../

cd swin224_mask
python swin224_mask.py
cd ../

cd swin224_pseudo.py
python swin224_pseudo.py
cd ../

cd swin224_mask_pseudo.py
python swin224_mask_pseudo.py
cd ../

#swintransformer 384
cd swin384
python swin224.py
cd ../

cd swin384_mask
python swin224_mask.py
cd ../

cd swin384_pseudo.py
python swin224_pseudo.py
cd ../

cd swin384_mask_pseudo.py
python swin224_mask_pseudo.py
cd ../