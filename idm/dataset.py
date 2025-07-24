import os
import torch
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import av
import bisect

from tqdm import tqdm


class DataSet(Dataset):
    """Dataset for inverse dynamics model

    Contents of Dataset:  If args.load_mp4==True, the data_file_path can be an mp4 file or an qpos file. 
    like:  {task_name}/f'episode_{episode_idx}.mp4', {task_name}/f'episode_{episode_idx}_qpos.pt', where:
    - the mp4 is saved as cv2.VideoWriter({task_name}/f'episode_{episode_idx}.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), fps=30, (width=640, height=720)). The length of the video is episode_len.
    - the 14-dim qpos of each trajectory is in the form of torch.zeros([episode_len, dim=14]).dtype(float32).cpu(), 
    saved with torch.save()
    - the json file is saved as {task_name}.json', which contains the information of the trajectory, but the caption
    is "Random Wing", not specfically processed yet (no use for IDM, but useful for video generation model).
    """
    
    def __init__(self, args, dataset_path, disable_pbar=False, type="train", preprocessor=None):
        self.data = []
        self.dataset_path = dataset_path
        self.type = type
        self.height = 720  # 480 + 240
        self.width = 640        
        self.video_bytes = []
        self.keyframe_indices = []
        self.video_metadata = []
        self.qpos_data = []
        self.video_lengths = []
        self.preprocessor = preprocessor
        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        for task_name in os.listdir(dataset_path):
            task_path = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_path):
                continue
            for file_name in tqdm(os.listdir(task_path), desc=f"Loading videos from {task_name}", disable=disable_pbar):
                if file_name.endswith('.mp4'):
                    episode_idx = file_name.split('_')[1].split('.')[0]
                    video_path = os.path.join(task_path, file_name)
                    qpos_path = os.path.join(task_path, f'episode_{episode_idx}_qpos.pt')
                    
                    if not os.path.exists(qpos_path):
                        print(f"Skipping {video_path} - no matching qpos file")
                        continue
                        
                    # Check video length without keeping capture open
                    cap = cv2.VideoCapture(video_path)
                    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Load qpos
                    qpos = torch.load(qpos_path)
                    
                    if video_length < 30:
                        print(f"Skipping {video_path} - too short")
                        continue
                        
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    with av.open(BytesIO(video_bytes)) as container:
                        stream = container.streams.video[0]

                        keyframes = []
                        for packet in container.demux(stream):
                            if packet.is_keyframe:
                                keyframes.append(packet.pts)

                        pts_per_frame = int((1 / stream.average_rate) / stream.time_base)
                        self.video_metadata.append({
                            'fps': float(stream.average_rate),
                            'width': stream.width,
                            'height': stream.height,
                            'time_base': stream.time_base,
                            'duration': stream.duration,
                            'pts_per_frame': pts_per_frame
                        })
                        self.keyframe_indices.append(keyframes)

                    self.video_bytes.append(video_bytes)
                    self.qpos_data.append(qpos)
                    self.video_lengths.append(video_length)
        
        self.data_begin = np.cumsum([0] + self.video_lengths[:-1])
        self.data_end = np.cumsum(self.video_lengths)

    def __len__(self):
        return self.data_end[-1]

    def __getitem__(self, idx):
        video_idx = np.searchsorted(self.data_end, idx, side='right')
        if video_idx < 0 or video_idx >= len(self.video_bytes):
            raise IndexError(f"Index {idx} out of bounds")
        
        if (video_idx >= len(self.keyframe_indices) or
            video_idx >= len(self.video_metadata)):
            raise RuntimeError(f"Metadata missing for video {video_idx}")
        
        local_idx = idx - self.data_begin[video_idx]
        
        video_bytes = BytesIO(self.video_bytes[video_idx])
        with av.open(video_bytes, format='mp4') as container:
            stream = container.streams.video[0]

            target_pts = local_idx * self.video_metadata[video_idx]['pts_per_frame']

            keyframes = self.keyframe_indices[video_idx]
            index = bisect.bisect_right(keyframes, target_pts) - 1
            nearest_keyframe = keyframes[max(index, 0)]

            container.seek(nearest_keyframe, stream=stream, backward=False, any_frame=False)

            for frame in container.decode(stream):
                if frame.pts >= target_pts:
                    frame_array = frame.to_ndarray(format='rgb24')
                    break
            else:
                raise RuntimeError("Frame not found")

        pos = self.qpos_data[video_idx][local_idx]
        image = Image.fromarray(frame_array)
        processed_image = self.preprocessor.process_image(image)
        return pos, processed_image