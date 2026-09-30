import numpy as np
import matplotlib.pyplot as plt

def displaySurfaceNormals(surfaceNormals):

    plt.figure()
    plt.subplot(131)
    plt.title('X')
    plt.imshow(surfaceNormals[:, :, 0])
    plt.colorbar()
    plt.axis('off')

    plt.subplot(132)
    plt.title('Y')
    plt.imshow(surfaceNormals[:, :, 1])
    plt.colorbar()
    plt.axis('off')

    plt.subplot(133)
    plt.title('Z')
    plt.imshow(surfaceNormals[:, :, 2])
    plt.colorbar()
    plt.axis('off')

    plt.show(block=False)
