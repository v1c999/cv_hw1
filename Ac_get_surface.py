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

   rows = surfaceNormals.shape[0]
   cols = surfaceNormals.shape[1]

   nx = surfaceNormals[:, :, 0]
   ny = surfaceNormals[:, :, 1]
   nz = surfaceNormals[:, :, 2]

   valid = np.abs(nz) > 1e-3           
   safe_nz = np.where(valid, nz, 1.0)
   p = np.where(valid, -nx / safe_nz, 0.0)   # dx/dz
   q = np.where(valid, -ny / safe_nz, 0.0)   # dy/dz

   def row_column():
      q0 = q.copy()
      q0[0, :] = 0
      top = np.cumsum(p[0, :])
      return top[None, :] + np.cumsum(q0, axis=0)

   def column_row():
      p0 = p.copy()
      p0[:, 0] = 0
      left = np.cumsum(q[:, 0])
      return left[:, None] + np.cumsum(p0, axis=1)

   def random_paths(n_paths):
      h_array = np.zeros((rows, cols))
      for k in range(n_paths):
         temp = np.zeros((rows, cols))
         for j in range(rows):
            for i in range(cols):
              if i == 0 and j == 0:
                 continue
              elif j == 0:
                 temp[j, i] = temp[j, i - 1] + p[j, i]
              elif i == 0:
                 temp[j, i] = temp[j - 1, i] + q[j, i]
              else:
                 if np.random.rand() < 0.5:
                    temp[j, i] = temp[j, i - 1] + p[j, i]
                 else:
                    temp[j, i] = temp[j - 1, i] + q[j, i]
         h_array += temp
      h_array /= n_paths
      return h_array

   if method == "row-column":
      heightMap = row_column()
   elif method == "column-row":
      heightMap = column_row()
   elif method == "average":
      heightMap = (row_column() + column_row())/2
   elif method == "random":
      heightMap = random_paths(10)
   else:
      raise ValueError("Unknown")

   return heightMap - heightMap.min()

   raise NotImplementedError("You should implement this.")
