# Author: Tao Hu <taohu620@gmail.com>
import math
import numpy as np
from transforms import transform_preds

def final_preds(heatmap, coords, center, scale, output_shape):
    #heatmap: 2958*16*64*64
    #coords: 2958*16*2
    # pose-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = heatmap[n,p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 0 and px < output_shape[0]-1 and py > 0 and py < output_shape[1]-1:
                diff = np.array([hm[px-1,py] - hm[px +1,py], hm[px,py-1] - hm[px][py+1]])
                coords[n,p,0] += np.sign(diff).tolist()[0] * .5 # add .25 or minus .25
                coords[n,p,1] += np.sign(diff).tolist()[1] * .5 #TODO bugger

    #coords += 0.5
    preds = np.copy(coords)

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], output_shape)

    return preds