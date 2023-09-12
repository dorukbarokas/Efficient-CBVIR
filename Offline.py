import glob
import os
import numpy as np
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import UnidentifiedImageError
import annoy
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set CUDA_VISIBLE_DEVICES to an empty string
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sys.path.append(os.path.dirname(os.path.abspath('featureextraction/solar/solar_global/')))
sys.path.append(os.path.dirname(os.path.abspath('KeyFrameExtraction/SBD')))
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor/main.py')))

from nearestneighbor.main import nns

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define batch size
batch_size = 16  # You can adjust this based on your GPU memory and image sizesö


# Load MobileNetV2 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()
# print(model)



def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = transform(img)
        return img
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error opening image file: {image_path}. Skipping...")
        return None

def extract_features(input_images):
    features_list = []

    # Create a DataLoader to handle batching
    data_loader = torch.utils.data.DataLoader(input_images, batch_size=batch_size, shuffle=False)

    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            # features = model(batch)
            features = model.forward_features(batch)
        features_list.append(features)

    # Concatenate features from all batches
    if len(features_list) > 0:
        all_features = torch.cat(features_list, dim=0)
        return all_features
    else:
        return None

def offline_phase(input_videos, save_path):
    start_time = time.time()
    print('>>> Initiating Offline Phase')
    empty_features = np.zeros((len(input_videos), 1280))
    # save_path = "mobilenetv2_features.npy"


    # Load and preprocess frame images
    frame_images = [preprocess_image(input_video) for input_video in input_videos]
    frame_images = [img for img in frame_images if img is not None]
    frame_images = torch.stack(frame_images)

    frame_images = frame_images.to(device).contiguous()
    frame_features_list = model(frame_images)
    frame_features_list = F.avg_pool2d(frame_features_list, kernel_size=frame_features_list.size()[2:])

    frame_features_list = frame_features_list.squeeze()
    # Convert PyTorch tensors to feature vectors (NumPy arrays)
    frame_features_list = frame_features_list.to("cpu").detach().numpy()

    # Extract the folder name from the input_videos path
    folder_name = os.path.basename(os.path.dirname(input_videos[0]))
    
    # Create a list of dictionaries where each dictionary contains the filename and feature vector
    data_to_save = []
    for input_video, features in zip(input_videos, frame_features_list):
        data_to_save.append({
            'filename': input_video,
            'features': features.tolist(),  # Convert NumPy array to a Python list
            'solar_features': np.zeros(1).tolist() 
        })

    # Set save_path to 'folder_name_features.json'
    save_path = f"{folder_name}_features.json"

    # Save the data to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(data_to_save, json_file)


    # Process any remaining images in the last batch
    if  frame_features_list is not None:
        print("Frame Features Shape:", frame_features_list.shape)
    else:
        print("No features extracted.")


    # Perform nearest neighbor search
    # print('>>> Performing Nearest Neighbour Search')
    # idx, dist, _ = nns(frame_features_list, query_features, 'faiss_flat_cpu')
    idx, dist, _ = nns(frame_features_list, empty_features, folder_name, 'annoy',  build=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print('>>> Done performing Nearest Neighbour Search')
    # print("Dist1", dist)
    print(f'>>> Offline stage took {elapsed_time} seconds')


    
    return frame_features_list

if __name__ == '__main__':
    input_videos = glob.glob(r"/space/dbarokasprofet/Hist_keyframes/*.jpg")
    save_path = "_features.json"  # Specify the name of the JSON file
    frame_features_list = offline_phase(input_videos, save_path)
    num_input_videos = len(input_videos)
    print(f'Number of input frames: {num_input_videos}')
    print(f'Features and filenames saved')
