from PIL import Image
from filter_core import sobel

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
r = sobel(gray_img)

r.show()




