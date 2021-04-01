import colorsys

import torchvision
import torchvision.transforms.functional as transforms
from PIL import Image
import numpy as np

import util
from util import round32

device = "cpu"

def load_image(path, size, scale=1):
    img = Image.open(path).convert(mode="RGB")
    size *= scale
    wpercent = size / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    size = round32(size)
    hsize = round32(hsize)

    return img.resize((int(size), hsize), Image.ANTIALIAS)

if __name__ == "__main__":
    size = 1024
    imgCont = load_image("content/city-large.jpg", size)
    imgOut = Image.open("output/green-paint-large_city-large_strength_0.075_scale_0.333_1024.png")

    height = imgOut.size[0]
    width = imgOut.size[1]

    imgCont_sequence = imgCont.getdata()
    imgCont_array = np.array(imgCont_sequence)

    imgOut_sequence = imgOut.getdata()
    imgOut_array = np.array(imgOut_sequence)

    Rcont = imgCont_array[:, 0]
    print(Rcont)
    print(Rcont.shape)
    Gcont = imgCont_array[:, 1]
    Bcont = imgCont_array[:, 2]
    Rout = imgOut_array[:, 0]
    Gout = imgOut_array[:, 1]
    Bout = imgOut_array[:, 2]

    Hcont = np.zeros([height, width])
    Scont = np.zeros([height, width])
    Lcont = np.zeros([height, width])
    Hout = np.zeros([height, width])
    Sout = np.zeros([height, width])
    Lout = np.zeros([height, width])

    output = np.zeros([height, width, 3])
    for i in range(height):
        for j in range(width):
            pixelI = i*width + j
            Hcont[i][j], Lcont[i][j], Scont[i][j] = colorsys.rgb_to_hls(Rcont[pixelI], Gcont[pixelI], Bcont[pixelI])
            Hout[i][j], Lout[i][j], Sout[i][j] = colorsys.rgb_to_hls(Rout[pixelI], Gout[pixelI], Bout[pixelI])
            output[i][j][0], output[i][j][1], output[i][j][2] = colorsys.hls_to_rgb(Hcont[i][j], Lout[i][j], Scont[i][j])
    print(output)

    img = Image.fromarray(output, 'RGB')
    img.save('output.png')








