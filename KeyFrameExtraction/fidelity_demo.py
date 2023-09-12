from fidelity import *
from main import *


if __name__ == '__main__':
    print(sys.argv[1])
    path = sys.argv[1]

    [hogs, hists, fdnorm, histnorm] = fidelity_descriptors(path)

    KE_method = "VSUMM_combi"
    performSBD = False
    presample = True
    keyframes_data, keyframe_indices, video_fps = keyframe_extraction(sys.argv[1], KE_method, performSBD, presample)
    save_keyframes(keyframe_indices, keyframes_data, KE_method)

    fid = fidelity(keyframe_indices, path, hists, hogs, fdnorm, histnorm)
    print("Fidelity: " + str(fid))