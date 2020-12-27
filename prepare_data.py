from PIL import Image
from os.path import join
from os import listdir
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data

DATA_PATH = "dataset"

"""
Div2k DataSet Reference
@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
} 
Type: DataSet
Link: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""
# Licence to BSD300
"""
InProceedings{MartinFTM01,
  author = {D. Martin and C. Fowlkes and D. Tal and J. Malik},
  title = {A Database of Human Segmented Natural Images and its
           Application to Evaluating Segmentation Algorithms and
           Measuring Ecological Statistics},
  booktitle = {Proc. 8th Int'l Conf. Computer Vision},
  year = {2001},
  month = {July},
  volume = {2},
  pages = {416--423}
}
"""

class DatasetLoader(data.Dataset):
    def __init__(self, dir, upscale_f):
        self.dir = join(DATA_PATH, dir)
        # The cropped size of the image
        self.reshape_size = 256 - (256 % upscale_f)
        super(DatasetLoader, self).__init__()
        self.set_img()
        # Compose several transform together
        self.input, self.target_trans = self.compose_transform(upscale_f)

    def compose_transform(self, upscale_f):
        """Initialize the input_transform and target transform for the data loader"""
        aug_lst = [CenterCrop(self.reshape_size), ToTensor()]
        transform_lst = []
        for t in range(2):
            compose_transform = aug_lst[:]
            if t == 0:
                compose_transform.append(Resize(self.reshape_size // upscale_f))
            transform_lst.append(Compose(compose_transform))
        return transform_lst[0], transform_lst[1]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert('YCbCr').split()[0]
        output_lst = []
        for t in [self.input, self.target_trans]:
            output_lst.append(t(img))
        return output_lst[0], output_lst[1]

    def set_img(self):
        """Initialize the img array"""
        self.img = []
        for m in listdir(self.dir):
            if self.check_if_img(m):
                self.img.append(join(self.dir, m))

    def check_if_img(self, filename):
        """Check if the img is a valid picture"""
        for format in [".png", ".jpg", ".jpeg"]:
            if filename.endswith(format):
                return True
        return False
