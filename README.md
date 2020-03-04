# WILDTRACK-toolkit
Tools accompanying the [WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) multi-camera dataset.

## 1. WILDTRACK annotations viewer
Graphical User Interface (GUI) illustration of the *WILDTRACK* dataset annotations. The script `annotations_viewer.py` draws the annotations for each of the views and displays the synchronized frames simultaneously. 
Input arguments (or symbolic links as the defaults): 
1. `dir_annotations`. Directory which contains the JSON multi-view annotation files. Each JSON file lists the annotations of a single multi-view frame, as provided when downloading the dataset.
2. `dir_frames`. The root directory which contains subdirectories. Each such subdirectory contains the frames per camera (as provided when downloading the dataset). For example:
    ```
    frames/
    ├── view0/
    ...
    └── view6/
    ```
It warns the user if less or more then 7 subdirectories (views) are given. The script works if there exists at least one subdirectory in `dir_frames`. Provides basic functionality such as, go to the previous/next multi-view frame with step 1 or 10.
 Run: `python annotations_viewer.py --help` for details.

### Dependencies
- Python 3.6 or later
- Tkinter
- Python JSON (JavaScript Object Notation) module
- OpenCV

#### Installing dependencies 
Please ensure you have the latest Python and NumPy. We recommend the Anaconda package manager.
```sh
$ conda create --name wildtrack-toolbox
$ source activate wildtrack-toolbox
$ conda install -c anaconda opencv=3.4.1  # might take a while
$ conda install -c anaconda pillow
```
Use `$ source deactivate` when done.

## 2. Camera calibration - example code
Code that uses the calibration files of the WILDTRACK dataset, and the OpenCV library.
`intersecting_area.py` draws the area considered for annotating the persons, in the WILDTRACK dataset, which lies approximately in the intersection between the fields of view of the seven cameras. Primarily, it generates 3D points on the ground plane (z-axis is 0) using a grid of size 1440x480, origin at (-300,  -90,    0) in centimeters (cm), with a step/offset of 2.5 cm for both the axis. It then projects these points in each view. Finally, it stores the marked frames in the given output directory. 
 Run: `python intersecting_area.py --help` for details.

### Dependencies
- Python 3.6 or later
- OpenCV
- Python ElementTree package

## Publication
*WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection*. *Tatjana Chavdarova, Pierre Baqué, Stéphane Bouquet, Andrii Maksai, Cijo Jose, Timur Bagautdinov, Louis Lettry, Pascal Fua, Luc Van Gool, François Fleuret*. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 5030-5039
<http://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html>


## License
Copyright (c) 2008 [Idiap Research Institute](http://www.idiap.ch/).

> *GNU GENERAL PUBLIC LICENSE*, Version 3, 29 June 2007.
WILDTRACK toolkit is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. WILDTRACK toolkit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with WILDTRACK toolkit. If not, see <http://www.gnu.org/licenses/>.


## See also
- [Multi Camera Calibration Suite](https://github.com/idiap/multicamera-calibration) 
