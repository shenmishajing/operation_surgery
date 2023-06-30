import os

import cv2
from tqdm import tqdm

# 读取mp4文件
cap = cv2.VideoCapture(
    "/data2/share/Datasets/operation_surgery/FromRunRunShaw/机器学习用视频文件夹/2023 胰腺手术/9242385 祝和松 whipple切成开放重建 2023-02-15/祝和松/CHN3_4.mp4"
)

# 确定输出文件夹
output_folder = "/data2/share/Datasets/image_classification/operation_surgery/images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 逐帧读取并保存图像
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with tqdm(total=total_frames) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        output_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.PNG")
        cv2.imwrite(output_filename, frame)
        pbar.update(1)

# 释放资源
cap.release()
