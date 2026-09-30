import numpy as np
import random
from tqdm import tqdm

def getSurface(surfaceNormals, method):
   """
   Integrate the estimated normals in surfaceNormals to compute the estimated heights of the surface.
   method can be one of four selections:
   "row-column": integrate along the rows, then the columns
   "column-row": integrate along the columns, then the rows
   "average": average the first two options
   "random": take the average of multiple (we recommend 10) paths
   """
   raise NotImplementedError("You should implement this.")
