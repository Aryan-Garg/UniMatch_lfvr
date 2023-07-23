#!/usr/bin/env python

import os
import glob
from tqdm.auto import tqdm 


def getAllDisp4pixel():
    left = glob.glob('/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data/*')
    right = glob.glob('/data2/raghav/datasets/Pixel4_3DP/rectified/A/Video_data/*')
    
    pbar_left = tqdm(left)
    for dirL in pbar_left:
        dirR = dirL.replace('B', 'A')
        videoName = dirL.split('/')[-1]
        pbar_left.set_description(f"Processing {videoName}")
        # print(videoName)

        # Call main_stereo.py with params from gmstereo_demo.sh
        os.system(f"CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
                    --inference_dir_left {dirL} \
                    --inference_dir_right {dirR} \
                    --inference_size 512 768 \
                    --output_path output/disp_pixel4_BA/{videoName} \
                    --resume pretrained/mixdata.pth \
                    --padding_factor 32 \
                    --upsample_factor 4 \
                    --num_scales 2 \
                    --attn_type self_swin2d_cross_swin1d \
                    --attn_splits_list 2 8 \
                    --corr_radius_list -1 4 \
                    --prop_radius_list -1 1 \
                    --reg_refine \
                    --num_reg_refine 3")
        

if __name__ == '__main__':
    getAllDisp4pixel()