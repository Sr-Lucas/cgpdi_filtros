from PIL import Image
from filter_core import kNearestNeightborFilter

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")


# Display the original image
gray_img.show()

# Filtro
cw = kNearestNeightborFilter(gray_img, 4)

# Display the negative image
cw.show()





