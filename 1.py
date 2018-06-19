import openface as op
import cv2
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title


a = op.AlignDlib(
    '/Users/aaron/anaconda3/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')


imgPath = '/Users/aaron/Desktop/1.jpg'
bgrImg = cv2.imread(imgPath)
rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

bb = a.getLargestFaceBoundingBox(rgbImg)
print(a.findLandmarks(rgbImg, bb=bb))
