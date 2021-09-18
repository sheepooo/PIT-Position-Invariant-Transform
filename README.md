## [PIT: Position-Invariant Transform for Cross-FoV Domain Adaptation (ICCV 2021)](https://arxiv.org/abs/2108.07142)

### Getting started
***
##### Clone the repo:

```
git clone https://github.com/sheepooo/pit-Position-Invariant-Transform/
cd pit-Position-Invariant-Transform
```

##### Requirements:

* Python >= 3.6 (numpy, itertools, argparse)
* pytorch >= 0.4.1

##### To test the PIT function, you can run:
```
python PIT_tensor.py
```
The images in "test_images" folder will be PITed and RPITed, and be saved in the same folder.


##### To PIT all images in a folder, you can run:
```
python pit_images_in_root_folder.py --fovx 'YourFovx' --root_path 'YourImageFolderName'
```
either fovx or fovy is enough (both is ok, too).
**NOTICE:** this code would change the images in the root folder directly, so you may need to back up the original images.

##### To PIT annotations for object detection (XML file in "Pascal VOC" format, as shown in the "test_annotations" folder), you can run:
```
python pit_annotations.py --fovx 'YourFovx' --root_path 'YourAnnotationFolderName'
```
either fovx or fovy is enough (both is ok, too).
**NOTICE:** this code would change the annotations in the root folder directly, so you may need to back up the original annotations.
