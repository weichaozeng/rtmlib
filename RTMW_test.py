import os
import time
import cv2
import argparse
from rtmlib import Body, draw_skeleton, draw_bbox
from tqdm import tqdm



if __name__ == '__main__':
    device = 'cuda'
    backend = 'onnxruntime'  # opencv, onnxruntime

    paser = argparse.ArgumentParser()
    paser.add_argument('--dataset_name', type=str, default='DexYCB_40')
    paser.add_argument('--dataset_dir', type=str, default='/home/zvc/Data/DexYCB/bop/data/')
    paser.add_argument('--video_dir', type=str, default='rgb', help='rgb | img | None')
    paser.add_argument('--save_root', type=str, default='vis_output/')

    args = paser.parse_args()
    save_dir = os.path.join(args.save_root, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640, 640),
        # # rtmw-x
        # pose='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip',
        # pose_input_size=(288, 384),
        # dwpose-l
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip',
        pose_input_size=(288, 384),
        backend=backend,
        device=device
    )

    for i, seq_name in tqdm(enumerate(sorted(os.listdir(args.dataset_dir)))):
        if args.dataset_name == "DexYCB_40" and int(seq_name) % 200 != 0:
            continue
        if args.dataset_name == "H2O-ego_19" and i % 10 != 0:
            continue
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

        first_frame = cv2.imread(img_files[0])
        img_h, img_w = first_frame.shape[:2]
        video_output_path = os.path.join(save_dir, f'{seq_name}.mp4')
        video_writer = cv2.VideoWriter(video_output_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_w, img_h)) 
        for img_file in img_files:
            img_name = os.path.basename(img_file)
            frame = cv2.imread(img_file)
            # s = time.time()
            keypoints, scores, bboxes = body(frame)
            # det_time = time.time() - s
            # print('det: ', det_time)

            img_show = frame.copy()

            # if you want to use black background instead of original image,
            # img_show = np.zeros(img_show.shape, dtype=np.uint8)

            img_show = draw_bbox(img_show,
                                bboxes,
                                color=(0, 255, 0)
                                )
            img_show = draw_skeleton(img_show,
                                    keypoints,
                                    scores,
                                    openpose_skeleton=openpose_skeleton,
                                    kpt_thr=0.43
                                    )

            video_writer.write(img_show)
        video_writer.release()
        print(f'Saved visualization to {video_output_path}')