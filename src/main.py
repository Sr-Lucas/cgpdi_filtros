from PIL import Image
from filter_core import expansion

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
r = expansion(gray_img, 3, 2)

r.show()




