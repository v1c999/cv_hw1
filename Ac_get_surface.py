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

   valid = np.abs(nz) > 1e-3            # zero normals (shadow/background) are invalid
   safe_nz = np.where(valid, nz, 1.0)
   p = np.where(valid, -nx / safe_nz, 0.0)   # dz/dx (along columns)
   q = np.where(valid, -ny / safe_nz, 0.0)   # dz/dy (along rows)

   def row_column():
      # along row 0 using p, then down each column using q
      q0 = q.copy()
      q0[0, :] = 0
      top = np.cumsum(p[0, :])
      return top[None, :] + np.cumsum(q0, axis=0)

   def column_row():
      # down column 0 using q, then along each row using p
      p0 = p.copy()
      p0[:, 0] = 0
      left = np.cumsum(q[:, 0])
      return left[:, None] + np.cumsum(p0, axis=1)

   def random_paths(n_paths):
      rng = np.random.default_rng(0)
      total = np.zeros((rows, cols))
      for _ in range(n_paths):
         from_above = rng.random((rows, cols)) < 0.5
         z = np.zeros((rows, cols))
         for y in range(rows):
               for x in range(cols):
                  if y == 0 and x == 0:
                     continue
                  if y == 0:
                     z[y, x] = z[y, x - 1] + p[y, x]
                  elif x == 0:
                     z[y, x] = z[y - 1, x] + q[y, x]
                  elif from_above[y, x]:
                     z[y, x] = z[y - 1, x] + q[y, x]
                  else:
                     z[y, x] = z[y, x - 1] + p[y, x]
         total += z
      return total / n_paths

   if method == "row-column":
      heightMap = row_column()
   elif method == "column-row":
      heightMap = column_row()
   elif method == "average":
      heightMap = 0.5 * (row_column() + column_row())
   elif method == "random":
      heightMap = random_paths(10)
   else:
      raise ValueError("Unknown method: " + str(method))

   return heightMap - heightMap.min()

   raise NotImplementedError("You should implement this.")
