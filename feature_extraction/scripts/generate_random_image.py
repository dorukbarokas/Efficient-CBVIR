import numpy as np
from PIL import Image

img = np.random.rand(720, 1280, 3) * 255
img = Image.fromarray(img.astype('uint8')).convert('RGB')
img.save('random_image.jpg')
