# simple_faceswap
Simple face-changing program using opencv-python and dlib

## Preparation ##
* pip install opencv-python, dlib
* Download dlib face shape detector model data：[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)，And unzip it under the models folder

## Implementation steps ##
1. Use the shape_predictor_68_face_landmarks.dat model of dlib to obtain 68 face feature points of face image im1 and camera image im2.
2. According to the feature points obtained in the previous step, the face masks im1_mask and im2_mask of the two pictures are obtained.
3. Using 3 feature points out of 68 feature points, an affine transformation is performed on the face image im1 to align its face with the face in the camera picture to obtain the picture affine_im1.
4. The affine_im1_mask is also obtained by performing the same affine transformation on the im1_mask of the face image.
5. Union the mask im2_mask and mask affect_im1_mask to get union_mask.
6. The seamlessClone function in opencv is used to perform Poisson fusion of the affine_im1 and camera image im2 after the affine transformation, and the mask is union_mask to obtain the fusion image seamless_im.

## Change face effect ##
* Jay Chou's handsome photo:

![JayChou.png](./faces/JayChou.png)

* The face change effect using Jaylen’s face as the replacement face:

![seamless_im.png](./faces/seamless_im.png)

