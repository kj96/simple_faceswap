# -*- coding: utf-8 -*-

import os
import cv2
import dlib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

models_folder_path = os.path.join(here, 'models')  # Model save folder
faces_folder_path = os.path.join(here, 'faces')  # Face image save folder
predictor_path = os.path.join(models_folder_path, 'shape_predictor_68_face_landmarks.dat')  # Model path
image_face_path = os.path.join(faces_folder_path, 'JayChou.png')  # Face picture path

detector = dlib.get_frontal_face_detector()  # dlib's forward face detector
predictor = dlib.shape_predictor(predictor_path)  # dlib's face shape detector


def get_image_size(image):
    """
    Get image size (height, width)
    :param image: image
    :return: (Height, width)
    """
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image, face_detector, shape_predictor):
    """
    Get face signs, 68 feature points
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68 feature points
    """
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        return None
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks


def get_face_mask(image_size, face_landmarks):
    """
    Get face mask
    :param image_size: size of picture
    :param face_landmarks: 68 size of picture
    :return: image_mask, Mask picture
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=255)

    # mask = np.zeros(image_size, dtype=np.uint8)
    # points = cv2.convexHull(face_landmarks)  # Convex hull
    # cv2.fillConvexPoly(mask, points, color=255)
    return mask


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    """
    Get picture 1 Affine transformed picture
    :param image1: Picture 1, the picture to be affine transformed
    :param image2: Picture 2, as long as it is used to obtain the size of the picture, generate an affine transformation picture of the same size
    :param face_landmarks1: Face feature points in picture 1
    :param face_landmarks2: Face feature points in picture 2
    :return: Affine transformed picture
    """
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                               face_landmarks2[three_points_index].astype(np.float32))
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    """
    Get the coordinates of the center point of the mask
    :param image_mask: Mask picture
    :return: Mask center
    """
    image_mask_index = np.argwhere(image_mask > 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    """
    Get the union of two masked parts
    :param mask1: mask_image, Mask 1
    :param mask2: mask_image, Mask 2
    :return: Two masks cover the union of parts
    """
    mask = np.min([mask1, mask2], axis=0)  # Conceal partial union
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)  # Reduce the mask size
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)  # Fuzzy mask
    return mask


def skin_color_adjustment(im1, im2, mask=None):
    """
    Skin tone adjustment
    :param im1: Picture 1
    :param im2: Picture 2
    :param mask: Face mask. If it exists, use the average of the face part to find the skin color conversion coefficient; otherwise, use Gaussian blur to find the skin color conversion coefficient
    :return: Picture 1 adjusted to the color of Picture 2
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(im1, (im1_ksize, im1_ksize), 0).astype(np.float)
        im2_factor = cv2.GaussianBlur(im2, (im2_ksize, im2_ksize), 0).astype(np.float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask)
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask)
        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))

    im1 = np.clip((im1.astype(np.float) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return im1


def main():
    im1 = cv2.imread(image_face_path)  # face_image
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1, detector, predictor)  # 68_face_landmarks
    if landmarks1 is None:
        print('{}:No face detected'.format(image_face_path))
        exit(1)
    im1_size = get_image_size(im1)  # Face size
    im1_mask = get_face_mask(im1_size, landmarks1)  # Face map face mask

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, im2 = cam.read()  # camera_image
        landmarks2 = get_face_landmarks(im2, detector, predictor)  # 68_face_landmarks
        if landmarks2 is not None:
            im2_size = get_image_size(im2)  # Camera picture size
            im2_mask = get_face_mask(im2_size, landmarks2)  # Camera picture face mask

            affine_im1 = get_affine_image(im1, im2, landmarks1, landmarks2)  # im1 (face map) affine transformed picture
            affine_im1_mask = get_affine_image(im1_mask, im2, landmarks1, landmarks2)  # Face mask of im1 (face image) affine transformed picture

            union_mask = get_mask_union(im2_mask, affine_im1_mask)  # Mask merge

            # affine_im1_face_image = cv2.bitwise_and(affine_im1, affine_im1, mask=union_mask)  # im1 (face picture) face
            # im2_face_image = cv2.bitwise_and(im2, im2, mask=union_mask)  # im2 (camera picture) face
            # cv2.imshow('affine_im1_face_image', affine_im1_face_image)
            # cv2.imshow('im2_face_image', im2_face_image)

            affine_im1 = skin_color_adjustment(affine_im1, im2, mask=union_mask)  # Skin tone adjustment
            point = get_mask_center_point(affine_im1_mask)  # im1 (face image) the center point of the face mask of the affine transformed image
            seamless_im = cv2.seamlessClone(affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # Poisson fusion

            # cv2.imshow('affine_im1', affine_im1)
            # cv2.imshow('im2', im2)
            cv2.imshow('seamless_im', seamless_im)
        else:
            cv2.imshow('seamless_im', im2)
        if cv2.waitKey(1) == 27:  # Press Esc to exit
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
