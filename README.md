

## Module
prepare_data : prepare the training data\
augment_setting: set up the model augment\
test: Evaluate the performance on dataset\
trainer: train the model\
model: construct FSRCNN neural network model\
model_path.pth : The output model we trained

###Dataset
trained model: train & validate ---BSD300
old_photo_test: Test dataset of old photo
div_2k_test: Test dataset of Div2k


## Train the model
train:

```bash
$ python3 main.py 
```

## Testing Instruction

To test the output:

```bash
$ python3 test.py
```
### Test on single image
Input the path of the input test image, and runs the above bash.\
The output will be stored in output.jpg, \
 denoised output will be store in denoise.png.\
```

```
###Test on dataset
When running the above bash line, the output image will prompt one by one.\
To proceed, please close the image window, until the program finish.
and it will be stored in its target output_result directory.\
For example, if we want to evaluate the performance of div2k dataset,\
the output will be stored in the path
```
 dataset/div2k_test/output_result
```
the denoised output will be stored in the path.
```
 dataset/div2k_test/denoise_result
```
Note that the denoised output will not be printed during running.

### Licence for using Div2k
```
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
```
### Licence for using BSD300
```
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
}```
```
### Licence of old photo data
The testing data we collect from old photo are the results of Ziyu Wan's Team \
Old Photo Restoration via Deep Latent Space Translation, PAMI Under Review.\
Here's the reference
```
https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life
arxiv.org/abs/2004.09484
```