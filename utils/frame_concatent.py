import numpy as np
import cv2

def resize_with_padding(frame, width, height):
    # Get original dimensions
    h, w = frame.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(width/w, height/h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image while maintaining aspect ratio
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create black canvas of target size
    if frame.ndim == 3:
        result = np.zeros((height, width, frame.shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate padding
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    
    # Place the resized image in the center
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

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
    
    width = 0
    
    for i in range(len(frames)):
        width = max(width, frames[i].shape[1])
        width = max(width, frames[i].shape[0])
    
    for i in range(len(frames)):
        frames[i] = resize_with_padding(frames[i], width, width)
        
    while len(frames) < 6:
        frames.append(np.zeros_like(frames[0]))
    
    row1 = np.concatenate(frames[0:3], axis=1)
    row2 = np.concatenate(frames[3:6], axis=1)

    return np.concatenate([row1, row2], axis=0)