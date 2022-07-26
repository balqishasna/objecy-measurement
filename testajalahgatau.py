from PIL import Image
import PIL.ImageOps    

image = Image.open('input\lampuu.jpeg')

inverted_image = PIL.ImageOps.invert(image)

inverted_image.save('new_name.jpeg')
