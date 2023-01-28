from PIL import Image
from filter_core import hightBoost

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
r = hightBoost(gray_img)

r.show()




