import glob
import os
import sys
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
import time
import annoy
from collections import Counter
from sklearn.metrics import average_precision_score
sys.path.append(os.path.dirname(os.path.abspath('featureextraction/solar/solar_global/')))
sys.path.append(os.path.dirname(os.path.abspath('KeyFrameExtraction/SBD')))
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor/main.py')))
from featureextraction.solar.solar_global.utils.networks import load_network
from nearestneighbor.main import nns
from featureextraction.fe_main import extract_features_global
from Filtering import filtering_stage
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import gc
import json



def calculate_ap(gt_files, pred_files):
    y_true = np.zeros(len(pred_files))
    RD = len(gt_files)

    for i, pred_file in enumerate(pred_files):
        if any(pred_file in gt_file for gt_file in gt_files):
            y_true[i] = 1

    # Calculate the average precision
    ap_sum = 0
    num_correct = 0
    for i, is_correct in enumerate(y_true, 1):
        if is_correct:
            num_correct += 1
            ap_sum += num_correct / i

    ap = ap_sum / max(1, RD)
    return ap

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

def online(input_images, folder_name):
    # input_images = glob.glob(r"/space/dbarokasprofet/Rel_Hist/all_query/video_*.jpg")
    input_images = glob.glob(r"/space/dbarokasprofet/Rel_Hist/all_query/video_0_1_2.jpg")
    # input_images = glob.glob(r"/space/dbarokasprofet/Oxford5k/Oxford_Query/*.jpg")
    # input_images = glob.glob(r"/space/dbarokasprofet/Paris6k/query/*.jpg")

    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device =  torch.device("cpu")
    # Load MobileNetV2 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    empty_features = np.zeros((len(input_images), 1280))

    # Load and preprocess query image
    query_images = [preprocess_image(input_image) for input_image in input_images]
    query_images = [img for img in query_images if img is not None]
    query_images = torch.stack(query_images)

    # query_images = query_images.to(device).contiguous()
    query_images = query_images.to("cuda").contiguous()
    query_features = model(query_images)
    query_features = F.avg_pool2d(query_features, kernel_size=query_features.size()[2:])
    
    if len(query_images) != 1:
        query_features = query_features.squeeze()
        query_features = query_features.to("cpu").detach().numpy()
    else:
        query_features = query_features.to("cpu").detach().numpy()
        query_features = query_features.reshape(1, -1)

    # Load the JSON file
    json_file_path = f"{folder_name}_features.json"
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    frame_features_list = [item['features'] for item in data]
    frame_features_list = np.array(frame_features_list)
    input_videos = [item['filename'] for item in data]
    input_videos = np.array(input_videos)

    # Extract the folder name from the input_videos path
    folder_name = os.path.basename(os.path.dirname(input_videos[0]))

    # print("shape:", query_features.shape)


    idx, dist, _ = nns(frame_features_list, query_features, folder_name, 'annoy', build = False)
    num_query_images = len(input_images)
    keyframes_size = len(frame_features_list)
    scaling_factor = 100000
    
    threshold_value = num_query_images * keyframes_size / scaling_factor
    # threshold = np.percentile(dist[0], threshold_value * 100)
    threshold = 0.82

    # Process any remaining images in the last batch
    # if query_features is not None:
    #     print("Query Features Shape:", query_features.shape)
    # else:
    #     print("No features extracted.")

    filtered_indices = []
    filtered_names = []
    for i, distances in enumerate(dist):
        filtered_indices.append([idx[i][j] for j, d in enumerate(distances) if d < threshold])
    filtered_indices = list(set([idx for sublist in filtered_indices for idx in sublist]))
    # print( "Filtered_ind" , filtered_indices)
    filtered_videos = [input_videos[i] for i in filtered_indices]
    top_k = 10


    output_data = []

    for query_idx in range(len(input_images)):
        query_output = []

        # Relevant frames
        query_image_name = os.path.splitext(os.path.basename(input_images[query_idx]))[0]
        RD = 10

        for frame_idx, frame_dist in zip(idx[query_idx], dist[query_idx]):
            frame_path = input_videos[frame_idx]
            query_output.append((os.path.basename(frame_path), frame_dist))  # Get only video name, not full path
        query_output = sorted(query_output, key=lambda x: x[1])[:RD]

        # Create an array with only names of videos from query_output
        Pred_Files = [video_name for video_name, _ in query_output]

        # print(" ")
        # print(f"Query {query_idx + 1}: {input_images[query_idx]}")
        # for rank, (frame_path, frame_dist) in enumerate(query_output, 1):
        #     print(f"Rank {rank}: Frame: {frame_path}, Distance: {frame_dist}")

    print(">>> Number of Shortlisted Frames are",len(filtered_videos) ,"out of", len(input_videos))

    torch.cuda.empty_cache()
    cuda_device = 'cuda:0'
    # cuda_device = 'cpu'
    net = load_network(cuda_device, 'resnet101-solar-best.pth')
    net.mode = 'test'
    query_images = []
    frame_images = []
    similar_images = {}
    frame_features_list_SOLAR = []
    empty_features = np.zeros((len(input_images), 2048))
    res = []
    size = 1024

    for input_image in input_images:
        try:
            img = Image.open(input_image)
            img = img.convert('RGB')
            img = np.array(img)
            if img.shape[2] == 4:
                img = img[..., :3]  # disregard the alpha layer if it is present
            query_images.append(img)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error opening query image file: {input_image}. Skipping...")
    
    query_features = extract_features_global(cuda_device, query_images, net, size)
    query_features = query_features.numpy()
    query_features = np.ascontiguousarray(query_features, dtype=np.float32)


    # Prepare batches of frame images
    batch_size = 16
    num_frames = len(filtered_indices)
    # num_batches = (num_frames + batch_size - 1) // batch_size

    # Initialize a list to store video indices with zero vector SOLAR features among filtered_indices
    input_videos = [item['filename'] for item in data]
    input_videos = np.array(input_videos)

    new_solar_features_list = []
    # Initialize a list to store solar_features from non-zero_vector_indices
    stored_solar_features = []

    # Find indices of frames with zero "solar_features"
    zero_vector_indices = [i for i, index in enumerate(filtered_indices) if np.all(np.array(data[index]['solar_features']) == 0)]
    real_zero_indices = [filtered_indices[i] for i in zero_vector_indices]
    stored_indices = [i for i, index in enumerate(filtered_indices) if not np.all(np.array(data[index]['solar_features']) == 0)]
    real_stored_indices = [filtered_indices[i] for i in stored_indices]

    num_batches = (len(real_zero_indices) + batch_size - 1) // batch_size

    # print("Zero:", real_zero_indices)
    # print("Stores:", real_stored_indices)


    if zero_vector_indices:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_frames)

            batch_frame_images = [Image.open(input_videos[index]) for index in real_zero_indices[start_idx:end_idx]]
            batch_frame_images = [img.convert('RGB') for img in batch_frame_images]

            # Resize all images in the batch to a consistent size
            max_width = max(img.size[0] for img in batch_frame_images)
            max_height = max(img.size[1] for img in batch_frame_images)

            batch_frame_images = [img.resize((max_width, max_height)) for img in batch_frame_images]
            batch_frame_images = [np.array(img) for img in batch_frame_images]
            batch_frame_images = np.array(batch_frame_images)

            # Extract features for the batch of frame images
            batch_frame_features = extract_features_global(cuda_device, batch_frame_images, net, size)
            batch_frame_features = batch_frame_features.numpy()
            batch_frame_features = np.ascontiguousarray(batch_frame_features, dtype=np.float32)

            frame_features_list_SOLAR.append(batch_frame_features)

            # Add the batch 'solar_features' to the new_solar_features_list
            new_solar_features_list.extend(batch_frame_features.tolist())

            # Run garbage collection to free up memory after each iteration
            gc.collect()

        # Concatenate all batches of frame features into a single array
        frame_features_list_SOLAR = np.concatenate(frame_features_list_SOLAR, axis=0)
        # print("Shape of frame_features_list:", frame_features_list_SOLAR.shape)
        # print("Size of frame_features_list:", frame_features_list_SOLAR.size)

    for i in stored_indices:
        stored_solar_features.append(data[filtered_indices[i]]['solar_features'])

    # Convert the non-zero solar features list to a NumPy array
    store_solar_features = np.array(stored_solar_features, dtype=np.float32)
    updated_frame_features_list_SOLAR = []
    # print("Shape of store_solar_features:", store_solar_features.shape)
    # print("Shape of frame_features_list_SOLAR:", frame_features_list_SOLAR.shape)

    k = 0
    l = 0
    # Append the non-zero solar features to frame_features_list_SOLAR
    if len(frame_features_list_SOLAR) > 0:
        # frame_features_list_SOLAR = np.append(frame_features_list_SOLAR, stored_solar_features, axis=0)
        # frame_features_list_SOLAR = np.array(frame_features_list_SOLAR)
        # Append frame features from filtered_indices
        for i, index in enumerate(filtered_indices):
            if index in real_zero_indices:
                updated_frame_features_list_SOLAR.append(frame_features_list_SOLAR[k])
                k = k + 1
            elif index in real_stored_indices:
                updated_frame_features_list_SOLAR.append(store_solar_features[l])
                l = l + 1
        # Convert the list of frame features to a NumPy array
        frame_features_list_SOLAR = np.array(updated_frame_features_list_SOLAR, dtype=np.float32)
    else:
        frame_features_list_SOLAR = stored_solar_features
        frame_features_list_SOLAR = np.array(frame_features_list_SOLAR)

    if zero_vector_indices:
        # Update 'solar_features' in the loaded JSON data
        for i, index in enumerate(real_zero_indices):
            data[index]['solar_features'] = new_solar_features_list[i]

        # Save the updated data back to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)



    print('>>> Performing Nearest Neighbour Search')
    search_start_time = time.time()
    
    idx, dist, _ = nns(frame_features_list_SOLAR, query_features, folder_name, 'linear', build = False)
    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f'>>> Done performing Nearest Neighbour Search (Time: {search_time} seconds)')
    # Get the top 10 similar gallery images
    top_k = len(filtered_indices)

    for query_idx in range(len(input_images)):
        query_output = []

        # Relevant frames
        query_image_name = os.path.splitext(os.path.basename(input_images[query_idx]))[0]
        RD = 10

        for frame_idx, frame_dist in zip(idx[query_idx], dist[query_idx]):
            frame_path = filtered_videos[frame_idx]
            query_output.append((os.path.basename(frame_path), frame_dist))  # Get only video name, not full path
        query_output = sorted(query_output, key=lambda x: x[1])[:RD]
        Pred_Files = [video_name for video_name, _ in query_output]

        print(" ")
        print(f"Query {query_idx + 1}: {input_images[query_idx]}")
        for rank, (frame_path, frame_dist) in enumerate(query_output, 1):
            print(f"Rank {rank}: Frame: {frame_path}, Distance: {frame_dist}")


    end_time = time.time()
    execution_time = end_time - start_time
    print("Whole Process Time: ", execution_time)
    return res


if __name__ == '__main__':
    folder_name = "Hist_keyframes"
    online(True, folder_name)
