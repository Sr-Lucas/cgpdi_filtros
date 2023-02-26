from PIL import Image
from filter_core import laplaciano

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
resultImg = laplaciano(gray_img)

resultImg.show()




