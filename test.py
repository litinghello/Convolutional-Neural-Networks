
from PIL import Image
from io import BytesIO
import numpy as np
import os, random, base64, cv2
from gen_captcha import scan_files
from gen_captcha import image_name_start_i
from gen_captcha import image_name_end_i
np.set_printoptions(threshold=np.inf)


image_file = Image.open("image/0008_4000000152.jpg").convert('L')
image_file = np.array(image_file)
image_out = image_file.flatten()/255

print(image_out.shape[0])
'''cv2.imshow('img', cv2.resize(np.array(image_out).reshape(30, 70), (140, 60), interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()
#image_file = np.array(image_file)
#print(process_image(image_file))

'''
