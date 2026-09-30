import numpy as np

def alignChannels(img, max_shift):
    """
    Given an image, img, where the channels are misaligned, return both the 
    aligned img as well as the predicted shift values using either SSD or CS.
    The predicted shift values should be less than max_shift in absolute value for each axis. 
    """
    raise NotImplementedError("You should implement this.")

