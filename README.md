# Codebase for Efficient-CBVIR
Efficient Content-Based Image Retrieval from Videos using Compact Deep Learning Networks with Re-ranking


We have designed and implemented a video image-based query search engine that strikes a good balance between efficiency and accuracy. Users can submit any video for the gallery and any image they want to search for, and the engine will return similar frames in an input video database. The main components of the system are shown in the following diagram:


<img src="https://github.com/dorukbarokas/Efficient-CBVIR/blob/main/Picture3.png" width="520">
<img src="https://github.com/dorukbarokas/Efficient-CBVIR/blob/main/Picture4.png" width="520">

This work is a combination of three master's thesis projects. Please check out our theses via the following links:
- [x] Sinian Li (Keyframe Extraction): http://resolver.tudelft.nl/uuid:d16300c5-6988-4172-8c20-0e2dfff8949f
- [x] Doruk Barokas Profeta (Feature Extraction):

**Features**

- Efficient keyframe extraction from videos.
- Feature extraction using compact deep learning networks (MobileNetV2).
- Integration of approximate nearest-neighbor search (ANNOY) methods for rapid retrieval.
- Re-ranking module to enhance retrieval accuracy. (ResNet101 + SOLAR)

**Installation Guideline:**

To run the system, first clone the repository. To install the correct packages, change the directory to the folder that contains the environment.yml file, build the environment and install packages by using the following command:

```
conda env create -f environment.yml
```

Next, download the solar global model that can be found [here](https://imperialcollegelondon.box.com/shared/static/fznpeayct6btel2og2wjjgvqw0ziqnk4.pth). Move the downloaded model to the following folder: 

```
../Efficient-CBVIR/featureextraction/solar/data/networks/
``` 

tbd for UI

**User Guideline:**
tdb


