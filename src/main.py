from PIL import Image
from filter_core import expand_image_nn, expand_image_bilinear

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
r = expand_image_bilinear("./assets/images/teste1.bmp", 2)

resultImg = Image.open("./filtered/" + r)

resultImg.show()




