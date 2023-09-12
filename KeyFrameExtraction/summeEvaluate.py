import os
from summe import *
import numpy as np
import random
from main import *

''' PATHS ''' 
HOMEDATA='C:\\Users\\Robert\\bin\\SumMe\\GT'
HOMEVIDEOS='C:\\Users\\Robert\\bin\\SumMe\\videos'


if __name__ == "__main__":
    # Take a random video and create a random summary for it
    included_extenstions=['webm']
    videoList=[fn for fn in os.listdir(HOMEVIDEOS) if any([fn.endswith(ext) for ext in included_extenstions])]
    videoName = videoList[int(round(random.random()*24))]
    print(videoName)
    videoName = 'Playing_on_water_slide.webm'
    videoName=videoName.split('.')[0] # take only the name of the video without extension
    video_path = HOMEVIDEOS + str("\\") + videoName + ".mp4"

    #In this example we need to do this to now how long the summary selection needs to be
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    nFrames=gt_data.get('nFrames')
    #print(nFrames[0][0])
    
    '''Example summary vector''' 
    #selected frames set to n (where n is the rank of selection) and the rest to 0
    summary_selections=np.random.random((nFrames[0][0],1))*20
    sumsel = np.zeros([nFrames[0][0],1])
    val_percentile = np.percentile(summary_selections,85)
    #print(val_percentile)
    #print(summary_selections)
    for i in range(0, len(summary_selections)):
        if summary_selections[i] >= val_percentile:
            sumsel[i] = 1
    print(nFrames[0][0])

    sumsels = {}
    f_measures = np.zeros([nFrames[0][0],1])
    summary_lengths = np.zeros([nFrames[0][0],1])
    for i in range(0, 6):
        keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(video_path, 0.1+i, 0.85)
        frame_count = nFrames[0][0]
        sumsels[i] = changeIdxFormat(keyframe_indices, frame_count)
        #sumsels[i] = random_summary(80+i, nFrames[0][0])
        [f_measures[i], summary_lengths[i]] = evaluateSummary(sumsels[i], videoName, HOMEDATA)

    '''plotting'''
    methodNames={'Sampling'};
    plotAllResults(sumsels,methodNames, f_measures, summary_lengths, videoName,HOMEDATA)
