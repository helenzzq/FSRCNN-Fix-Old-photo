import torch
from os import listdir
from os.path import join
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math
from augment_setting import Augment
from skimage.metrics import structural_similarity
from pathlib import Path


def output_img(output_data, cb, cr, outputs):
    """ Output the desired image"""
    output_data = output_data.cpu()
    y = (output_data.data[0].numpy() * 255.0).clip(0, 255)
    ycbr_aug = [Image.fromarray(np.uint8(y[0]), mode='L'), cb, cr]
    resize(ycbr_aug)
    out_img = Image.merge('YCbCr', ycbr_aug).convert('RGB')
    out_img.save(outputs)


def set_model(y):
    """Set up the model"""
    device_arg = "cpu"
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device_arg = "cuda"
    div = torch.device(device_arg)
    model = torch.load(args.model, map_location=lambda storage, loc: storage).to(div)
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0]).to(div)
    return model(data)


def denoise(is_Grey=False):
    """Denoise the output image"""
    input_img = cv2.imread('output.jpg')
    if not is_Grey:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    result = cv2.fastNlMeansDenoising(input_img, None, 3, 7, 21)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig('denoise.png', bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.show()


def resize(ycbr):
    """Resize the image"""
    for i in range(len(ycbr)):
        if i != 0:
            ycbr[i] = ycbr[i].resize(ycbr[0].size, Image.BICUBIC)


def get_psnr(original, output):
    """Calculate PSNR of the image output"""
    mse = np.mean((original - output) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def get_ssim(original, output):
    """Get the SSIM of the image output"""
    grey_origin = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grey_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    ssim = structural_similarity(grey_origin, grey_output, full=True)[0]
    return ssim


def test_pic_group(test_old_photo):
    """Test all the picture and return the psnr and SSIM of
    denoised image and raw output image"""
    # Test on old photo dataset
    if test_old_photo:
        hr_dir = 'dataset/old_photo_test/hr_old/'
        testing_dir = 'dataset/old_photo_test/lr_old/'
        output_dir = 'dataset/old_photo_test/output_result/'
        denoise_dir = 'dataset/old_photo_test/denoise_result/'
    # Test on Div2k dataset
    else:
        hr_dir = 'dataset/div2k_test/hr_div2k_test/'
        testing_dir = 'dataset/div2k_test/lr_div2k_test/'
        output_dir = 'dataset/div2k_test/output_result/'
        denoise_dir = 'dataset/div2k_test/denoise_result/'

    input_name = listdir(testing_dir)
    avg_psnr, avg_ssim = 0, 0
    avg_psnr_d, avg_ssim_d = 0, 0
    for i in range(len(input_name)):
        name = join(testing_dir, input_name[i])
        # Check if filename is a valid image
        if name.endswith(".png"):
            input_img = Image.open(name)
            output_path = join(output_dir, input_name[i][:-4] + "_" + "result.jpg")
            print(name)
            test_single_pic(input_img, "output.jpg")
            # Apply denoise to the output image
            denoise(test_old_photo)
            output_imgs = cv2.imread("output.jpg")
            hr_img = cv2.imread(join(hr_dir, input_name[i]))
            # show_output()
            # upscale the denoised image
            img = Image.open('denoise.png')
            img = img.resize((hr_img.shape[1], hr_img.shape[0]), Image.ANTIALIAS)
            # move the output image and denoised image to desired directory
            img.save('denoise.png')
            denoise_img = cv2.imread("denoise.png")
            denoise_pth = join(denoise_dir, input_name[i][:-4] + "_" + "result.jpg")
            Path("denoise.png").rename(denoise_pth)
            # Calculate the average PSNR and SSIM
            avg_psnr += get_psnr(hr_img, output_imgs)
            avg_ssim += get_ssim(hr_img, output_imgs)
            Path("output.jpg").rename(output_path)
            avg_psnr_d += get_psnr(hr_img, denoise_img)
            avg_ssim_d += get_ssim(hr_img, denoise_img)
    avg_psnr = avg_psnr / (len(input_name) - 1)
    avg_ssim = avg_ssim / (len(input_name) - 1)
    avg_psnr_d = avg_psnr_d / (len(input_name) - 1)
    avg_ssim_d = avg_ssim_d / (len(input_name) - 1)
    print("The output PSNR is {},SSIM is {}".format(avg_psnr, avg_ssim))
    print("The denoised output PSNR is {}, SSIM is {}".format(avg_psnr_d, avg_ssim_d))
    return avg_psnr, avg_ssim


def show_output():
    """Show the output"""
    output = Image.open("output.jpg")
    plt.imshow(output)
    plt.show()


def test_single_pic(input_img, output):
    """Test single picture with original HR image and no need to downscale"""
    img = input_img.convert('YCbCr')
    y, cb, cr = img.split()
    model = set_model(y)
    output_img(model, cb, cr, output)


def single_pic_testing():
    """Testing single pic"""
    input_img = Image.open('dataset/div2k_test/lr_div2k_test/0058.png')
    test_single_pic(input_img, "output.jpg")
    hr_img = cv2.imread('dataset/div2k_test/hr_div2k_test/0058.png')
    output = cv2.imread('output.jpg')
    print("The psnr is {}".format(get_psnr(hr_img, output)))
    print("The SSIM is {}".format(get_ssim(hr_img, output)))


if __name__ == '__main__':
    # Augment Setting

    # print(input_img.shape)
    aug = Augment()
    aug.set_test_augment()
    args = aug.parser.parse_args()

    # Comment out this to test on dataset, input true to test on div2k dataset
    test_pic_group(test_old_photo=False)

    # Comment out this to test on single image
    # single_pic_testing()
    # denoise()
