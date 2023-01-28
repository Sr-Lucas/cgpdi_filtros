from PIL import Image
from filter_core import getImghistogram, equalizateImage

img = Image.open("./assets/images/teste8.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
eq = equalizateImage(gray_img)
getImghistogram(gray_img, 'teste8')
getImghistogram(eq, 'teste8eq')

# Display the negative image
eq.show()





