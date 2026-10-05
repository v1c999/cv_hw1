import numpy as np

def photometricStereo(imarray, lightdirs):
    """
    Given the normalized images in imarray and the corresponding light source directions
    lightdirs, calculate the albedo and normals of the surface depicted. You may need to use np.linalg.lstsq.
    """
    images = np.asarray(imarray)
    L = np.array(lightdirs)
    m, h, w = images.shape
    I = images.reshape(m, -1)
    print(L.shape, I.shape)

    G, *_ = np.linalg.lstsq(L, I, rcond=None)

    albedo = np.linalg.norm(G, axis=0)
    normals = G / np.maximum(np.linalg.norm(G, axis=0), 1e-8)

    albedo_img = albedo.reshape(h, w)
    normals_img = normals.T.reshape(h, w, 3)
    
    return albedo_img, normals_img

    raise NotImplementedError("You should implement this.")
