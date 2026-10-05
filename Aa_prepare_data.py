import numpy as np

def prepareData(imArray, ambientImage):
    """
    Take in an array of images in imArray and subtract the ambientImage from each one.
    Then renormalize the images by setting negative values to zero and rescaling the 
    maximum intensity of the image collection to 1 (use the max value across all images).
    """

    print(imArray.shape)
    print(ambientImage.shape)

    max_val = 0
    imArray = np.moveaxis(np.asarray(imArray), -1, 0)
    #print(imArray.shape)
    ambientImage = np.asarray(ambientImage, dtype=np.float64)
    #print(ambientImage.shape)

    for image in imArray:
        image -= ambientImage
        image[image < 0 ] = 0
        if np.max(image) > max_val:
            max_val = np.max(image)

    for image in imArray:
        image /= max_val

    return imArray


    raise NotImplementedError("You should implement this.")
