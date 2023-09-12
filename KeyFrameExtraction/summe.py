'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Package to evaluate and plot summarization results
% on the SumMe dataset
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''
import scipy.io
import warnings
import numpy as np
import matplotlib.pyplot as plt

def evaluateSummary(summary_selection,videoName,HOMEDATA):

     '''Evaluates a summary for video videoName (where HOMEDATA points to the ground truth file)   
     f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2013 
     NOTE: This is only a minimal version of the matlab script'''
     # Load GT file
     gt_file=HOMEDATA+'/'+videoName+'.mat'
     gt_data = scipy.io.loadmat(gt_file)
     #print(len(list(summary_selection)))
     user_score=gt_data.get('user_score')
     nFrames=user_score.shape[0] #rows
     #print(nFrames)
     nbOfUsers=user_score.shape[1] # 15
    
     # Check inputs
     if len(summary_selection) < nFrames:
          print("error1")
          warnings.warn('Pad selection with %d zeros!' % (nFrames-len(summary_selection)))
          summary_selection.extend(np.zeros(nFrames-len(summary_selection)))

     elif len(summary_selection) > nFrames:
          print("error2")
          warnings.warn('Crop selection (%d frames) to GT length' %(len(summary_selection)-nFrames))
          summary_selection=summary_selection[0:nFrames];
             
     
     # Compute pairwise f-measure, summary length and recall
     #print(summary_selection)
     summary_indicator = summary_selection
     #print(summary_indicator.shape)
     #summary_indicator= np.where(summary_selection > 0, summary_selection, 1)
     #print(summary_indicator)
     #summary_indicator=np.array(map(lambda x: (1 if x>0 else 0),summary_selection));
     user_intersection=np.zeros((nbOfUsers,1)); #15 zeros
     user_union=np.zeros((nbOfUsers,1)); #15 zeros
     user_length=np.zeros((nbOfUsers,1)); #15 zeros
     user_score_normalized = np.where(user_score > 0, 1, 0)
     #print(user_score_normalized.shape)
     for userIdx in range(0,nbOfUsers):
         #gt_indicator=np.array(map(lambda x: (1 if x>0 else 0),user_score[:,userIdx]))
         gt_indicator = user_score_normalized[:, userIdx]
         gt_indicator = gt_indicator.reshape(len(gt_indicator), 1)
         #user_intersection[userIdx]=np.sum(gt_indicator*summary_indicator);
         user_intersection[userIdx] = np.sum(np.multiply(gt_indicator, summary_indicator))
         union = gt_indicator + summary_indicator
         user_union = np.where(union > 0, 1, 0)
         #user_union[userIdx]=sum(np.array(map(lambda x: (1 if x>0 else 0),gt_indicator + summary_indicator)));
         user_length[userIdx]=sum(gt_indicator)
    
     recall=user_intersection/user_length
     p=user_intersection/np.sum(summary_indicator) #precision
     #print(p)
     #print(recall)

     f_measure=[]
     for idx in range(0,len(p)):
          if p[idx]>0 or recall[idx]>0:
               f_measure.append(2*recall[idx]*p[idx]/(recall[idx]+p[idx]))
          else:
               f_measure.append(0)
     nn_f_meas=np.max(f_measure);
     f_measure=np.mean(f_measure);
    
     nnz_idx=np.nonzero(summary_selection)
     nbNNZ=len(nnz_idx[0])
         
     summary_length=float(nbNNZ)/float(len(summary_selection));
       
     recall=np.mean(recall);
     p=np.mean(p);
     
     return f_measure, summary_length


def plotAllResults(summary_selections,methods,f_measures,summary_lengths,videoName,HOMEDATA):
    '''Evaluates a summary for video videoName and plots the results
      (where HOMEDATA points to the ground truth file) 
      NOTE: This is only a minimal version of the matlab script'''
    
    # Get GT data
    gt_file=HOMEDATA+'/'+videoName+'.mat'
    gt_data = scipy.io.loadmat(gt_file)
    user_score=gt_data.get('user_score')
    nFrames=user_score.shape[0]
    nbOfUsers=user_score.shape[1]

    ''' Get automated summary score for all methods '''
    automated_fmeasure={};
    automated_length={};
    for methodIdx in range(0,len(methods)):
        #summaryIndices=np.sort(np.unique(summary_selections[methodIdx]))
        automated_fmeasure[methodIdx]=np.zeros(len(summary_selections));
        automated_length[methodIdx]=np.zeros(len(summary_selections));
        idx=0
        automated_fmeasure[methodIdx][0] = f_measures[methodIdx]
        automated_length[methodIdx][0] = summary_lengths[methodIdx]
        #for selIdx in summaryIndices:
            #if selIdx>0:
                #curSummary=np.array(map(lambda x: (1 if x>=selIdx else 0),summary_selections[methodIdx]))
                #f_m, s_l = evaluateSummary(curSummary,videoName,HOMEDATA)
                #automated_fmeasure[methodIdx][idx]=f_m
                #automated_length[methodIdx][idx]=s_l
                #idx=idx+1

    print("hoe")
    ''' Compute human score '''
    human_f_measures=np.zeros(nbOfUsers)
    human_summary_length=np.zeros(nbOfUsers)
    for userIdx in range(0, nbOfUsers):
        user_selection = user_score[:,userIdx]
        user_selection =  user_selection.reshape(len(user_selection), 1)
        human_f_measures[userIdx], human_summary_length[userIdx] = evaluateSummary(user_selection,videoName,HOMEDATA);


    avg_human_f=np.mean(human_f_measures)
    avg_human_len=np.mean(human_summary_length)


    ''' Plot results'''
    fig = plt.figure()

    colors=['r','g','m','c','y']
    for methodIdx in range(0,len(methods)):
        #p2=plt.plot(100*automated_length[methodIdx],automated_fmeasure[methodIdx],'-'+colors[methodIdx])
        print(f_measures)
        print(summary_lengths[10])
        p2 = plt.plot(summary_lengths*100, f_measures, '-'+colors[0])
        #p2=plt.scatter(15,f_measures[0])

    p1=plt.scatter(100*human_summary_length,human_f_measures)
    plt.xlabel('summary length[%]')
    plt.ylabel('f-measure')
    plt.title('f-measure for video '+videoName)

    legend=list(methods)
    legend.extend(['individual humans'])
    plt.legend(legend)
    plt.ylim([0,0.85])
    plt.xlim([0,20])
    plt.plot([5, 5],[0, 1],'--k')
    plt.plot([15.1, 15.1],[ 0, 1],'--k')
    plt.show()
