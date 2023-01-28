import math
from time import time
from scipy import stats as st
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

MASK_SIZE = 3

def negativeFilter(img):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
        # Get pixel value at (x,y) position of the image
        pixel = img.getpixel((i,j))

        # Invert color
        outputPixel = 255 - pixel
        copy.putpixel((i,j), outputPixel)
  
  return copy



def logFilter(img):
  copy = img.copy()

  c = 255/math.log(255+1, 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(logTransform(c, pixel))
      copy.putpixel((i,j),(outputPixel))
  
  return copy

def logTransform(c, pixel):
    return c * math.log(float(1 + pixel), 10)



def inverseLogFilter(img):
  copy = img.copy()

  c = 255/((255+1) ** 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(invLogTransform(c, pixel))
      copy.putpixel((i,j),(outputPixel))
  
  return copy

def invLogTransform(c, pixel):
  return c * (float(1 + pixel) ** 10)


def nthPoewerFilter(img, gamma = 1.0):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      pixel = img.getpixel((i,j))

      outputPixel: float = 255 * ((pixel / 255) ** gamma)
      copy.putpixel((i,j),(outputPixel.__ceil__()))
  
  return copy



def nthRootFilter(img, gamma = 1.0):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      pixel = img.getpixel((i,j))

      outputPixel = 255 * ((pixel / 255) ** (1/gamma))
      copy.putpixel((i,j),(outputPixel.__ceil__()))
  
  return copy


def horizontalMirroringFilter(img: Image):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - x, y)))
  
  return imgCopy


def verticalMirroringFilter(img: Image):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel((x, (img.size[0]-1) - y)))
  
  return imgCopy

# G(x, y) = F(y, (TAM - 1) - x)
def rotation90clockwise(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel((y, (img.size[0]-1) - x)))
  
  return imgCopy

# G(x, y) = F((TAM - 1) - y, x)
def rotation90anticlockwise(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - y, x)))
  
  return imgCopy

# G(x, y) = F((TAM - 1) - x, (TAM - 1) - y)
def rotation180(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - x, (img.size[0]-1) - y)))
  
  return imgCopy



def compression(img, a = 3, b = 1):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      pixel = img.getpixel((x, y))
      rPixel = (pixel / a) - b
      imgCopy.putpixel((x, y), rPixel.__ceil__())
  
  return imgCopy


def expansion(img, a = 3, b = 1):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      pixel = img.getpixel((x, y))
      rPixel = (a * pixel) - b
      imgCopy.putpixel((x, y), rPixel.__ceil__())
  
  return imgCopy

def fillMask(img, x, y):
  mask = np.zeros((3, 3))

  mask[0][0] = img.getpixel((x-1, y-1))
  mask[0][1] = img.getpixel((x, y-1))
  mask[0][2] = img.getpixel((x+1, y-1))
  mask[1][0] = img.getpixel((x-1, y))
  mask[1][1] = img.getpixel((x, y))
  mask[1][2] = img.getpixel((x+1, y))
  mask[2][0] = img.getpixel((x-1, y+1))
  mask[2][1] = img.getpixel((x, y+1))
  mask[2][2] = img.getpixel((x+1, y+1))
  
  return mask


def maxFilter(img):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      max = 0
      for a in range(0, 3):
        for b in range(0, 3):
          if (mask[a][b] > max):
            max = mask[a][b]
      
      imgCopy.putpixel((x, y), max.__ceil__())
  
  return imgCopy


def minFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      min = 255
      for a in range(0, 3):
        for b in range(0, 3):
          if (mask[a][b] < min):
            min = mask[a][b]
      
      imgCopy.putpixel((x, y), min.__ceil__())
  
  return imgCopy



def modaFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      modeResult = st.mode(mask.flatten())

      imgCopy.putpixel((x, y), modeResult.mode[0].__ceil__())
  
  return imgCopy


def pseudoMedianaFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)
      L = MASK_SIZE * MASK_SIZE
      M = int((L + 1)/2)

      maskFlattened = mask.flatten()

      arr = []
      i = 0
      while M+i <= L:
        arr.append([])
        for k in range(i, M+i):
          arr[i].append(maskFlattened[k])
        i += 1
      
      maxes = []
      mins = []
      for i in range(0, arr.__len__()-1):
        maxes.append(max(arr[i]))
        mins.append(min(arr[i]))
      
      minmax = min(maxes)
      maxmin = max(mins)

      pseudoMedian = (maxmin + minmax)/2

      imgCopy.putpixel((x, y), pseudoMedian.__ceil__())
  
  return imgCopy



def NNRAmpliation(img, size):
  imgCopy = img.copy()

  return imgCopy.resize((size, size), Image.Resampling.NEAREST)


def BIRAmpliation(img, size):
  imgCopy = img.copy()

  return imgCopy.resize((size, size), Image.Resampling.BILINEAR)


def kNearestNeightborFilter(img, k):
  if k >= (MASK_SIZE * MASK_SIZE) - 1:
    return 0

  imgCopy = img.copy()
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      middleX = (MASK_SIZE/2).__floor__()
      middleY = (MASK_SIZE/2).__floor__()

      pivo = mask[middleX][middleY]
    
      maskFlatten = mask.flatten()
      maskFlatten[int(mask.size/2)] = 99999999

      nearCalcArr = []
      for i in range(0, mask.size):
        nearCalcArr.append(abs(maskFlatten[i] - pivo))
      
      sortedArr = nearCalcArr.copy()
      sortedArr.sort()

      sum = 0
      for i in range(0, k):
        index = nearCalcArr.index(sortedArr[i])
        sum += maskFlatten[index]
      
      knnResult = sum/k

      imgCopy.putpixel((x, y), knnResult.__ceil__())

  return imgCopy


def getImghistogram(img, imgName):  
  ih = img.size[0]-1
  iw = img.size[1]-1

  hist = np.zeros([256], np.int32)

  for x in range(0, ih):
    for y in range(0, iw):
      hist[img.getpixel((y, x))] += 1
  
  plt.figure()
  plt.title('Gray Scale Histogram')
  plt.xlabel('Intensity Level')
  plt.ylabel('Intesity Frequency')
  plt.xlim([0, 256])
  plt.plot(hist)
  plt.savefig(f"./hist/{imgName}.jpg")


def equalizateImage(img):
  imgArr = np.asarray(img)
  hist = np.bincount(imgArr.flatten(), minlength=256)

  numPixels = np.sum(hist)
  hist = hist/numPixels

  chistogramArr = np.cumsum(hist)

  transforMap = np.floor(255 * chistogramArr).astype(np.uint8)

  imgList = list(imgArr.flatten())

  eqImgList = [transforMap[p] for p in imgList]

  eqImgArr = np.reshape(np.asarray(eqImgList), imgArr.shape)

  equalizedImage = Image.fromarray(eqImgArr, mode='L')

  return equalizedImage


def sumImages(img1Path: str, img2Path: str, img1percentual = 0.5, img2percentual = 0.5):
  img1 = cv2.imread(img1Path)
  img2 = cv2.imread(img2Path)
  dst = cv2.addWeighted(img1, img1percentual, img2, img2percentual, 0)

  ext = img1Path.split('/').pop().split('.')[1]
  filename = str(time()).replace('.', '') + "." + ext

  cv2.imwrite(f"./filtered/{filename}", dst)

  return f"{filename}.{ext}"


def laplaciano(img):
  # 2x NC central - NC Esquerdo - NC Direito

  imgCopy = img.copy()
    
  mask = np.zeros([3])
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask[0] = img.getpixel((x-1, y))
      mask[1] = img.getpixel((x, y))
      mask[2] = img.getpixel((x+1, y))

      pixelR = (2 * mask[1]) - mask[0] - mask[2]

      imgCopy.putpixel((x, y), pixelR.__ceil__())
    
  return imgCopy


def hightBoost(img):
  # (2x NC central - NC Esquerdo - NC Direito) + NC Central

  imgCopy = img.copy()
    
  mask = np.zeros([3])
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask[0] = img.getpixel((x-1, y))
      mask[1] = img.getpixel((x, y))
      mask[2] = img.getpixel((x+1, y))

      pixelR = ((2 * mask[1]) - mask[0] - mask[2]) + mask[1]

      imgCopy.putpixel((x, y), pixelR.__ceil__())
    
  return imgCopy