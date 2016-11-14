import cv2
import math
import numpy as np
import warnings

def auto_canny(image, sigma=0.33):
    import cv2
    import numpy as np

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged    # return the edged image
	
def compute_skew(image):
    #image = cv2.bitwise_not(image)
    height, width = image.shape
    edges = auto_canny(image)
    lines = cv2.HoughLinesP(edges.copy(), 1, np.pi/180, 100, minLineLength=width / 8.0, maxLineGap=10)
    angle = 0.0
    if lines == None:
        return angle,edges
    else:
        nlines = lines.size
        no_lines = nlines/4
        for i in range(no_lines):
            for x1, y1, x2, y2 in lines[i]:
                angle += np.arctan2(y2 - y1,x2 - x1)
        angle = (angle / no_lines)
        return  angle, edges


def deskew(image, angle):
    angle = np.math.degrees(angle)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows,cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (cols, rows), center)


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def largest_rotated_rect(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  #print w,h
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  #print sin_a,cos_a
  if side_short <= 2.*sin_a*cos_a*side_long:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr
	
def SkewDetectAndCorrect(img):
    rows,cols = img.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angle,edges = compute_skew(img.copy())

	if angle == 0:
            return img,edges
	else:
            deskewed_image = deskew(img.copy(), angle)
##          deskewed_image = cv2.resize(deskewed_image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            rows,cols = deskewed_image.shape
            deskewed_cropped = crop_around_center(deskewed_image,*largest_rotated_rect(cols,rows,angle))
            edges = auto_canny(deskewed_cropped)
            return deskewed_cropped, edges
	










