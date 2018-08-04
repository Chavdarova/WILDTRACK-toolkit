#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Shows the intersecting area between the seven cameras of the WILDTRACK dataset,
which is the area considered for annotating the persons.
This script generates 3D points on the ground plane (z-axis is 0) using a grid
of size 1440x480, origin at (-300,  -90,    0) in centimeters (cm), and uses
step of 2.5 cm in both the axis. It then projects these points in each of the
views. Finally, it stores one frame per each view where the grid is shown.
The purpose of this code is to demonstrate how the provided calibration files
can be used, using the OpenCV library.

For information regarding the WILDTRACK dataset, see the following paper:
'WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection',
Tatjana Chavdarova, Pierre Baqué, Stéphane Bouquet, Andrii Maksai, Cijo Jose,
Timur Bagautdinov, Louis Lettry, Pascal Fua, Luc Van Gool, François Fleuret;
In Proceedings of The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), 2018, pp. 5030-5039. Available at: http://openaccess.thecvf.com/
content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html

To download the dataset visit: https://cvlab.epfl.ch/data/wildtrack

Copyright (c) 2008 Idiap Research Institute, http://www.idiap.ch/
Written by Tatjana Chavdarova <tatjana.chavdarova@idiap.ch>

This file is part of WILDTRACK toolkit.

WILDTRACK toolkit is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

WILDTRACK toolkit is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with WILDTRACK toolkit. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import argparse
import numpy as np
import cv2
from xml.dom import minidom
from os import listdir
from os.path import isfile, isdir, join, split, dirname, exists
import xml.etree.ElementTree as ElementTree

# specific to the WILDTRACK dataset:
_grid_sizes = (1440, 480)
_grid_origin = (-300, -90, 0)
_grid_step = 2.5


def load_opencv_xml(filename, element_name, dtype='float32'):
    """
    Loads particular element from a given OpenCV XML file.

    Raises:
        FileNotFoundError: the given file cannot be read/found
        UnicodeDecodeError: if error occurs while decoding the file

    :param filename: [str] name of the OpenCV XML file
    :param element_name: [str] element in the file
    :param dtype: [str] type of element, default: 'float32'
    :return: [numpy.ndarray] the value of the element_name
    """
    if not isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        tree = ElementTree.parse(filename)
        rows = int(tree.find(element_name).find('rows').text)
        cols = int(tree.find(element_name).find('cols').text)
        return np.fromstring(tree.find(element_name).find('data').text,
                             dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(e)
        raise UnicodeDecodeError('Error while decoding file %s.' % filename)


def parse_args():
    """
    Parses the input command line arguments, and if the given output
    directories does not exist, it creates it.

    :return: [argparse.Namespace]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_list", type=str, default="lists/frames.list",
                        help="list of image directories, default: %(default)s")
    parser.add_argument("--intrinsic_list", type=str,
                        default="lists/intrinsic_calibration_zeroDist.list",
                        help="list of intrinsic calibration files, default: %(default)s")
    parser.add_argument("--extrinsic_list", type=str, default="lists/extrinsic_calibration.list",
                        help="list of extrinsic calibration files, default: %(default)s")
    parser.add_argument("--img_prefix", type=str, default="intersecting_area/grid",
                        help="prefix for the output images, default: %(default)s")
    parser.add_argument('--fr_ext', type=str, default='.png',
                        help='extension of the frames, default: %(default)s')
    _args = parser.parse_args()
    # Create the output directory if it does not exist:
    if dirname(_args.img_prefix) != "" and not exists(dirname(_args.img_prefix)):
        os.makedirs(dirname(_args.img_prefix))
    return _args


def _load_content_lines(_file):
    """
    Loads the content of _file & returns it as a list of lines.

    Raises:
        FileNotFoundError: the given file is not found
        ValueError: the file is found, but it is empty

    :param _file: [str] path to a file
    :return: [list of str]
    """
    if not isfile(_file):
        raise FileNotFoundError("File %s not found." % _file)
    _lines = open(_file).read().splitlines()
    if len(_lines) == 0:
        raise ValueError("File %s is empty." % _file)
    return _lines


def _load_imagas(_dirs, _n=0, _ext='png'):
    """
    Loads the _n-th image of each of the given directories.

    Raises:
        NotADirectoryError: a string in _dirs is not a directory
        IndexError: Found fewer files then _n in a directory

    :param _dirs: [list of str] list of directories
    :param _n: [int, optional] which image to be loaded, default: 0
    :param _ext: [str, optional] extension of the file/image, default: 'png'
    :return: [list of numpy.ndarray] loaded images
    """
    _imgs = []
    for _, _dir in enumerate(_dirs):
        if not isdir(_dir):
            raise NotADirectoryError('%s is not a directory.' % _dir)

        files = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f)) and f.endswith(_ext)]
        if len(files) <= _n:
            raise IndexError("Found fewer files in %s than selected: %d" % (_dir, _n))
        _imgs.append(cv2.imread(sorted(files)[_n]))
    return _imgs


def load_all_extrinsics(_lst_files):
    """
    Loads all the extrinsic files, listed in _lst_files.

    Raises:
        FileNotFoundError: see _load_content_lines
        ValueError: see _load_content_lines

    :param _lst_files: [str] path of a file listing all the extrinsic calibration files
    :return: tuple of ([2D array], [2D array]) where the first and the second integers
             are indexing the camera/file and the element of the corresponding vector,
             respectively. E.g. rvec[i][j], refers to the rvec for the i-th camera,
             and the j-th element of it (out of total 3)
    """
    extrinsic_files = _load_content_lines(_lst_files)
    rvec, tvec = [], []
    for _file in extrinsic_files:
        xmldoc = minidom.parse(_file)
        rvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
        tvec.append([float(number)
                     for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])
    return rvec, tvec


def project_grid_points(_origin, _size, _offset, rvec, tvec, camera_matrices, dist_coef):
    """
    Generates 3D points on a grid & projects them into all the views,
    using the given extrinsic and intrinsic calibration parameters.

    :param _origin: [tuple] of the grid origin (x, y, z)
    :param _size: [tuple] of the size (width, height) of the grid
    :param _offset: [float] step for the grid density
    :param rvec: [list] extrinsic parameters
    :param tvec: [list] extrinsic parameters
    :param camera_matrices: [list] intrinsic parameters
    :param dist_coef: [list] intrinsic parameters
    :return:
    """
    points = []
    for i in range(_size[0] * _size[1]):
        x = _origin[0] + _offset * (i % 480)
        y = _origin[1] + _offset * (i / 480)
        points.append(np.float32([[x, y, 0]]))  # ground points, z-axis is 0

    projected = []
    for c in range(len(camera_matrices)):
        imgpts, _ = cv2.projectPoints(np.asarray(points),  # 3D points
                                      np.asarray(rvec[c]),  # rotation rvec
                                      np.asarray(tvec[c]),  # translation tvec
                                      camera_matrices[c],  # camera matrix
                                      dist_coef[c])  # distortion coefficients
        projected.append(imgpts)
    return projected


def load_all_intrinsics(_lst_files):
    """
    Loads all the intrinsic files, listed in _lst_files.

    Raises:
        TypeError: input is not str
        FileNotFoundError: see _load_content_lines
        ValueError: see _load_content_lines
        FileNotFoundError: see load_opencv_xml
        UnicodeDecodeError: see load_opencv_xml

    :param _lst_files: [str] path of a file listing all the intrinsic calibration files
    :return: tuple (cameraMatrices[list], distortionCoef[list])
    """
    if not isinstance(_lst_files, str):
        raise TypeError(f"Type mismatch. Found {type(_lst_files)}, expected str.")
    intrinsic_files = _load_content_lines(_lst_files)
    _cameraMatrices, _distCoeffs = [], []
    for _file in intrinsic_files:
        _cameraMatrices.append(load_opencv_xml(_file, 'camera_matrix'))
        _distCoeffs.append(load_opencv_xml(_file, 'distortion_coefficients'))
    return _cameraMatrices, _distCoeffs


def draw_points(images, points):
    """
    Draws the 2D points in each of the images.
    The images are modified in-place.

    Raises:
        TypeError: either images or points is not list
        ValueError: the first dimension of the input does not match

    :param images: [list of numpy.ndarray] list of images
    :param points: [list of numpy.ndarray] list of 2D points
    :return: [None]
    """
    if not isinstance(images, list):
        raise TypeError(f"Type mismatch. Found {type(images)}, expected list.")
    if not isinstance(points, list):
        raise TypeError(f"Type mismatch. Found {type(points)}, expected list.")
    if not len(images) == len(points):
        raise ValueError("Length mismatch: %d and %d" % (len(images), len(points)))
    for v in range(_n_views):
        for p in range(len(projected[v])):
            try:
                if (points[v][p].ravel())[0] >= 0 and (points[v][p].ravel())[1] >= 0:
                    cv2.circle(images[v], tuple(points[v][p].ravel()), 3, (255, 0, 0), -1)  # Blue
            except OverflowError:
                pass


if __name__ == '__main__':
    args = parse_args()
    _folders = _load_content_lines(args.folder_list)
    frames = _load_imagas(_folders, _n=0, _ext=args.fr_ext)

    rvec, tvec = load_all_extrinsics(args.extrinsic_list)
    cameraMatrices, distCoeffs = load_all_intrinsics(args.intrinsic_list)

    assert len(frames) == len(rvec) == len(tvec), "Inconsistent number of views"
    _n_views = len(frames)

    projected = project_grid_points(_grid_origin, _grid_sizes, _grid_step,
                                    rvec, tvec, cameraMatrices, distCoeffs)
    draw_points(frames, projected)

    for v in range(_n_views):
        cv2.imwrite(args.img_prefix + str(v + 1) + '.png', frames[v])
    print(f"Images stored in {dirname(args.img_prefix)}")
