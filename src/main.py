from PIL import Image
from filter_core import pseudoMedianaFilter

img = Image.open("./assets/images/teste1.bmp")
gray_img = img.convert("L")

# Display the original image
gray_img.show()

# Filtro
cw = pseudoMedianaFilter(gray_img)

# Display the negative image
cw.show()



