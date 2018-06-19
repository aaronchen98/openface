import openface as op
import cv2
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
from pylab import show


a = op.AlignDlib(
    '/Users/aaron/anaconda3/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')


imgPath = '/Users/aaron/Desktop/1.jpg'
bgrImg = cv2.imread(imgPath)
rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

bb = a.getLargestFaceBoundingBox(rgbImg)
points = a.findLandmarks(rgbImg, bb=bb)


im = array(Image.open('/Users/aaron/Desktop/1.jpg'))
imshow(im)

x = []
y = []
for item in points:
    x.append(item[0])
    y.append(item[1])


# 使用红色星状标记绘制点
plot(x, y, 'r*')
# 绘制连接前两个点的线
# plot(x[:2],y[:2])
# 添加标题，显示绘制的图像
title('Landmarks')
show()
