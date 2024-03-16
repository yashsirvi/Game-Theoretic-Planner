from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np
import cv2 as cv
import random
from scipy.interpolate import splprep,splev
from collections import deque
import numpy as np
from matplotlib import transforms

img = cv.imread('example_map1.png', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_CONSTANT, None, value = 0)
kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])*10
# img = cv.dilate(img, kernel*(-2))
img = cv.erode(img, kernel)
ret,thresh1 = cv.threshold(img,70,255,0)

thresh1 = thresh1/255

# img = invert(img)
skel1 = skeletonize(thresh1)
skeleton = skeletonize(thresh1)
n = np.array(skeleton.nonzero())
n = n.T
print(n.shape)


stack = deque()
randpt = random.randint(0,n.shape[0])
print(randpt)
mid_path = []
stack.append(n[randpt])
n = np.delete(n, randpt, axis=0)
count = 0
lim1 = n.shape[0]
while len(stack) > 0 and count < lim1:
  v = stack.pop()
  try:
    idxs = np.linalg.norm(n-v, axis=1).argmin()
  except ValueError:
    pass
  try:
    i = n[idxs]
  except:
    pass
  if skel1[i[0],i[1]] > 0:
    stack.append(i)
    count = count + 1
    mid_path.append(i)
    skel1[i[0],i[1]] = 0
    n = np.delete(n, idxs, axis=0)



mid_path = np.array(mid_path)
mid_path_t = mid_path.T
tck, u = splprep([mid_path_t[0],mid_path_t[1]], s=20000)
new_points = splev(u, tck)
new_points = np.array(new_points)

# plt.plot(new_points[0], new_points[1], 'r-')
print(mid_path_t.shape)

fig, ax = plt.subplots()
thresh2 = thresh1.T
im = ax.imshow(thresh2, cmap=plt.cm.gray)
ax.plot(new_points[0], new_points[1], 'r-')
plt.show()