from __future__ import print_function
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
from PIL import Image
from augment_setting import Augment

def output_img(output_data):
    output_data = output_data.cpu()
    y = (output_data.data[0].numpy() * 255.0).clip(0, 255)
    ycbr_aug = [Image.fromarray(np.uint8(y[0]), mode='L'), cb, cr]
    resize(ycbr_aug)
    out_img = Image.merge('YCbCr', ycbr_aug).convert('RGB')
    out_img.save(args.output)

def set_model():
    device_arg = "cpu"
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device_arg = "cuda"
    div = torch.device(device_arg)
    model = torch.load(args.model, map_location=lambda storage, loc: storage).to(div)
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0]).to(div)
    return model(data)

def denoise():
    img1 = cv2.imread('output.jpg')
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img2, None, 3, 3, 7, 21)
    plt.imshow(dst)
    plt.show()


def resize(ycbr):
    for i in range(len(ycbr)):
        if i != 0:
            ycbr[i] = ycbr[i].resize(ycbr[0].size, Image.BICUBIC)

if __name__ == '__main__':

    # Augment Setting
    aug = Augment()
    aug.set_test_augment()
    args = aug.parser.parse_args()
    img = Image.open(args.input).convert('YCbCr')
    y, cb, cr = img.split()
    model = set_model()
    output_img(model)
    denoise()
