from PIL import Image
from filter_core import inverseLogFilter, logFilter, negativeFilter

img = Image.open("./assets/images/teste1.bmp")

# Display the original image
img.show()

# Filtro
negativeFilter(img)

# Display the negative image
img.show()


