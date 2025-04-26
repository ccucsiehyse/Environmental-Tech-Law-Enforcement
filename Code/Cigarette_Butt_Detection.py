#Run this code with TrackNetv3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *

import cv2


def generate_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return []
    
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If no frame is returned, the end of the video is reached
        if not ret:
            print("Reached the end of the video or no frame returned.")
            break
        
        frame_list.append(frame)
    
    # Release the video capture object
    cap.release()
    
    # Check if any frames were captured
    if len(frame_list) == 0:
        print(f"No frames captured from {video_file}")
    else:
        print(f"Captured {len(frame_list)} frames from {video_file}")
    
    return frame_list


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    args = parser.parse_args()

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    # Generate frames from video file
    frame_list = generate_frames(args.video_file)
    print(f"Shape of frame_list: {np.array(frame_list).shape}")

    if len(frame_list) == 0:
        print(f"Error: No frames captured from video {args.video_file}. Exiting.")
        exit(1)

    # Get dimensions from the first frame in the frame list
    h, w = frame_list[0].shape[:2]
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                          'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len

    dataset = Shuttlecock_Trajectory_Dataset(
        seq_len=seq_len,
        sliding_step=seq_len,
        data_mode='heatmap',
        bg_mode=bg_mode,
        frame_arr=np.array(frame_list)[:, :, :, ::-1],  # Convert frame format as needed
        padding=True
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().cuda()
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()

        # Predict
        tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

    # Write CSV file
    pred_dict = tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)

    print('Done.')
