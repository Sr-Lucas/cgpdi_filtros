import math


def negativeFilter(img):
  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
        # Get pixel value at (x,y) position of the image
        pixel = img.getpixel((i,j))

        # Invert color
        outputPixel = 255 - pixel
        img.putpixel((i,j), outputPixel)



def logFilter(img):
  c = 255/math.log(255+1, 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(logTransform(c, pixel))
      img.putpixel((i,j),(outputPixel))

def logTransform(c, pixel):
    return c * math.log(float(1 + pixel), 10)



def inverseLogFilter(img):
  c = 255/((255+1) ** 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(invLogTransform(c, pixel))
      img.putpixel((i,j),(outputPixel))

def invLogTransform(c, pixel):
  return c * (float(1 + pixel) ** 10)



  