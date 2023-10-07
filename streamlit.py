import streamlit as st
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

UPLOAD_FOLDER = './static/'
class_map = {0: 'laugh', 1: 'pullup', 2: 'punch', 3: 'pour', 4: 'pick'}

st.title("Video Classification App")

class Net3D(nn.Module):
    # Define the learnable parameters by calling the respective modules (nn.Conv2d, nn.MaxPool2d, etc.)
    def __init__(self):
        super(Net3D, self).__init__()

        # Calling conv3d module for convolution
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=50, kernel_size=2, stride=1)

        # Calling MaxPool3d module for max pooling with downsampling of 2
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)

        self.conv2 = nn.Conv3d(in_channels=50, out_channels=100, kernel_size=(1, 3, 3), stride=1)

        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(100*30*30, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 10)

    # Defining the structure of the network
    def forward(self, x):
        # Applying relu activation after each conv layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Reshaping to 1d for giving input to fully connected units
        x = x.view(-1, 100*30*30)

        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(x)
        return x

# Function to get frames from a video file
def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)

    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if fn in frame_list:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()

    if len(frames) > 16:
        extra_frames = len(frames) - 16
        frames = frames[:-extra_frames]

    return frames, v_len

# Function to preprocess frames
def preprocess_frames(frames):
    h, w = 128, 128
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    test_transformer = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    frames_tr = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame = test_transformer(frame)
        frames_tr.append(frame)

    return frames_tr

# Define the Streamlit app
def main():
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file:
        st.subheader("Uploaded Video:")
        st.video(uploaded_file)

        frames, _ = get_frames(uploaded_file, n_frames=16)
        random_frame = Image.fromarray(frames[5])
        frames_tr = preprocess_frames(frames)

        if len(frames_tr) > 0:
            frames_tr = torch.stack(frames_tr)

        model = Net3D()
        classification_model_path = "best_updated_model_path"

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(classification_model_path))
        else:
            model.load_state_dict(torch.load(classification_model_path, map_location=torch.device('cpu')))

        with torch.no_grad():
            outputs = model(frames_tr.view(-1, 16, 3, 128, 128))
            _, predicted = outputs.max(1)
            predicted_idx = predicted.item()
            predicted_class = class_map.get(predicted_idx, "Unknown")

        st.subheader("Random Frame from Video:")
        st.image(random_frame, caption="Random Frame", use_column_width=True)

        st.subheader("Predicted Class:")
        st.write(predicted_class)

if __name__ == '__main__':
    main()
