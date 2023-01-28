from PIL import Image
from filter_core import sumImages

img = Image.open("./assets/images/teste8.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
sumImages("./assets/images/teste1.bmp", "./assets/images/teste8.bmp")




