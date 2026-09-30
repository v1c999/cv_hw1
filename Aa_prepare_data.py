import numpy as np

def prepareData(imArray, ambientImage):
    """
    Take in an array of images in imArray and subtract the ambientImage from each one.
    Then renormalize the images by setting negative values to zero and rescaling the 
    maximum intensity of the image collection to 1 (use the max value across all images).
    """
    raise NotImplementedError("You should implement this.")
