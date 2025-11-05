import os
import time
import cv2
import argparse
from rtmlib import Body, draw_skeleton     
from tqdm import tqdm



if __name__ == '__main__':
    device = 'cuda'
    backend = 'onnxruntime'  # opencv, onnxruntime

    paser = argparse.ArgumentParser()
    paser.add_argument('--dataset_name', type=str, default='HO3D_v2_train')
    paser.add_argument('--dataset_dir', type=str, default='/home/zvc/Data/HO3D_v2/train/')
    paser.add_argument('--video_dir', type=str, default='', help='rgb | img | None')
    paser.add_argument('--save_root', type=str, default='vis_output/')

    args = paser.parse_args()
    save_dir = os.path.join(args.save_root, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
                det_input_size=(640, 640),
                pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
                pose_input_size=(288, 384),
                backend=backend,
                device=device)

    for seq_name in tqdm(os.listdir(args.dataset_dir)):
        seq_path = os.path.join(args.dataset_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue
        if args.video_dir != '':
            video_path = os.path.join(seq_path, args.video_dir)
        else:
            video_path = seq_path
        # print(f'Processing {seq_name}...')

        img_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path)
                            if f.endswith('.png') or f.endswith('.jpg')])
        if len(img_files) == 0:
            print(f'No image files found in {video_path}, skip.')
            continue    
        for img_file in img_files:
            img_name = os.path.basename(img_file)
            frame = cv2.imread(img_file)
            # s = time.time()
            keypoints, scores = body(frame)
            # det_time = time.time() - s
            # print('det: ', det_time)

            img_show = frame.copy()

            # if you want to use black background instead of original image,
            # img_show = np.zeros(img_show.shape, dtype=np.uint8)

            img_show = draw_skeleton(img_show,
                                    keypoints,
                                    scores,
                                    openpose_skeleton=openpose_skeleton,
                                    kpt_thr=0.43)

            save_path = os.path.join(save_dir, seq_name)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, img_name), img_show)