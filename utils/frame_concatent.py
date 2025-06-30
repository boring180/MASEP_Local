import numpy as np
import cv2

def concatent_frame(frames):
    """
    Concatenate six frames into a single frame.
    The order of the parts is:
    top left, top right, bottom left, bottom right, center left, center right
    
    Args:
        frames: a list of 6 frames
        
    Returns:
        a single frame
    """
    
    for i in range(len(frames)):
        if i == 0:
            continue
        else:
            frames[i] = cv2.resize(frames[i], (frames[0].shape[1], frames[0].shape[0]))
        
    while len(frames) < 6:
        frames.append(np.zeros_like(frames[0]))
    
    row1 = np.concatenate(frames[0:3], axis=1)
    row2 = np.concatenate(frames[3:6], axis=1)

    return np.concatenate([row1, row2], axis=0)