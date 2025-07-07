def slicing_frame(frame):
    """
    Slicing the frame into 6 parts, each part is a 1/3 width and 1/2 height of the original frame.
    The order of the parts is:
    top left, top right, bottom left, bottom right, center left, center right
    
    Args:
        frame: the frame to be sliced
        
    Returns:
        a list of 6 parts of the frame, 
    """
    width = frame.shape[1] // 3
    height = frame.shape[0] // 2
    frames = [frame[0:height, 0:width], frame[0:height, width:width*2], frame[0:height, width*2:width*3], frame[height:height*2, 0:width], frame[height:height*2, width:width*2], frame[height:height*2, width*2:width*3]]
    return frames
