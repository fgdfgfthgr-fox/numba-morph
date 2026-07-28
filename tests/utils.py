import scipy.ndimage as ndimage

def connectivity_to_footprint(connectivity):
    # 2D
    if connectivity == 4:
        footprint = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        footprint = ndimage.generate_binary_structure(2, 2)
    # 3D
    elif connectivity == 6:
        footprint = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        footprint = ndimage.generate_binary_structure(3, 2)
    else:
        footprint = ndimage.generate_binary_structure(3, 3)
    return footprint