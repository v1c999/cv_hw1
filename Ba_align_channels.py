import numpy as np

def alignChannels(img, max_shift):
    """
    Given an image, img, where the channels are misaligned, return both the 
    aligned img as well as the predicted shift values using either SSD or CS.
    The predicted shift values should be less than max_shift in absolute value
    for each axis. 
    """
    print(img.shape, img.max(), img.min())

    channel_1 = img[:, :, 0]
    channel_2 = img[:, :, 1]
    channel_3 = img[:, :, 2]

    dx_min, dx_max = -max_shift[0] + 1, max_shift[0] - 1
    dy_min, dy_max = -max_shift[1] + 1, max_shift[1] - 1

    ssd_2_min, ssd_3_min = np.inf, np.inf
    cs_2_max, cs_3_max = -np.inf, -np.inf
    best_shift_2, best_shift_3 = (0, 0), (0, 0)

    for i in range(dx_min, dx_max+1):
        for j in range(dy_min, dy_max+1):
            channel_2_shifted = np.roll(channel_2, shift=(i, j), axis=(0, 1))
            ssd_2 = np.sum((channel_1 - channel_2_shifted) ** 2)
            cs_2 = np.sum(channel_1 * channel_2_shifted) / (np.linalg.norm(channel_1) * np.linalg.norm(channel_2_shifted))
            if ssd_2 < ssd_2_min:
                ssd_2_min = ssd_2
                best_shift_2 = (i, j)
            """if cs_1 > cs_1_max:
                cs_1_max = cs_1
                best_shift_1 = (i, j)"""
            channel_3_shifted = np.roll(channel_3, shift=(i, j), axis=(0, 1))
            ssd_3 = np.sum((channel_1 - channel_3_shifted) ** 2)
            cs_3 = np.sum(channel_1 *channel_3_shifted) / (np.linalg.norm(channel_1) * np.linalg.norm(channel_3_shifted))
            if ssd_3 < ssd_3_min:
                ssd_3_min = ssd_3
                best_shift_3 = (i, j)
            """if cs_3 > cs_3_max:
                cs_3_max = cs_3
                best_shift_3 = (i, j)"""

    channel_2_shifted = np.roll(channel_2, shift=best_shift_2, axis=(0, 1))
    channel_3_shifted = np.roll(channel_3, shift=best_shift_3, axis=(0, 1))
    new_img = np.stack([channel_1, channel_2_shifted, channel_3_shifted], axis=2)
    img_shift = np.array([best_shift_2, best_shift_3])
    return new_img, img_shift

    #raise NotImplementedError("You should implement this.")

