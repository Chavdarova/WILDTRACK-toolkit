#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" 
Simple Graphical User Interface for displaying the annotations of the WILDTRACK dataset.
Given: 
    1. directory which contains the JSON multi-view annotation files (as provided);
    2. directory containing sub-directories, where each sub-directory contains the 
       frames per camera (as provided);
it draws the annotations for each of the views, and displays them simultaneously. 
Provides basic functionality such as, go to the previous/next frame with step 1 or 10.

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

from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import json
import argparse
import cv2
import math
import warnings


_N_VIEWS = 7  # specific to WILDTRACK (used  only for warning)


def read_json(filename):
    """
    Decodes a JSON file & returns its content.

    Raises:
        FileNotFoundError: file not found
        ValueError: failed to decode the JSON file
        TypeError: the type of decoded content differs from the expected (list of dictionaries)

    :param filename: [str] name of the JSON file
    :return: [list] list of the annotations
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        with open(filename, 'r') as _f:
            _data = json.load(_f)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode {filename}.")
    if not isinstance(_data, list):
        raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
    if len(_data) > 0 and not isinstance(_data[0], dict):
        raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
    return _data


def _subdirs(root_dir, _sort=True):
    """
    Returns a list of the sub-directories found in root_dir.

    Raises:
        NotADirectoryError: root_dir is not a directory

    :param root_dir: [str] root directory
    :param _sort: [bool] if True returns sorted list
    :return: [list of strings] sub-directories (absolute paths)
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError('%s is not a directory.' % root_dir)
    _sub_dirs = [os.path.join(root_dir, dir_name)
                 for dir_name in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, dir_name))]
    if _sort:
        _sub_dirs.sort()
    return _sub_dirs


def _files(root_dir, _extension, _sort=True):
    """
    Returns a list of the files found in root_dir with the given extension.

    Raises:
        NotADirectoryError: root_dir is not a directory

    :param root_dir: [str] root directory
    :param _extension: [str] extension of the files
    :param _sort: [bool] if True returns sorted list
    :return: [list of strings] sub-directories (absolute paths)
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError('%s is not a directory.' % root_dir)
    files = [os.path.join(root_dir, _f) for _f in os.listdir(root_dir)
             if os.path.isfile(os.path.join(root_dir, _f)) and _f.endswith(_extension)]
    if _sort:
        files.sort()
    return files


def init_window(window_name=None):
    """
    Initializes tkinter non-resizable window.
    :param window_name: [str, optional] Name of the window
    :return: [tkinter.Tk] object Tk, representing the main window
    """
    gui = Tk()
    gui.title(window_name)
    gui.resizable(False, False)
    return gui


def _on_close():
    """
    Helper function which displays a message to the user, 
    asking to confirm the closing of the window. 
    Used by close_window.
    :return: [None]
    """
    if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
        gui.destroy()


def close_window(_gui):
    """
    Closes the given tkinter.Tk window.
    :param _gui: [tkinter.Tk] window to be closed
    :return: [None]
    """
    _gui.protocol("WM_DELETE_WINDOW", _on_close)
    _gui.mainloop()


class AnnotationsViewer:
    """
    Implements GUI viewer of multi-camera annotations. 
    Given annotation files & frames, as well as a reference to tkinter window object, 
    it displays the annotations drawn on given frames.
    To link annotations with frames, it assumes these are named identically.
    For e.g. 0000.png and 0000.json would be the frame and annotation names, respectively.
    Provides basic functionality such as, go to the previous/next frame with step 1 or 10.
    """

    def __init__(self, _gui, _opt, _verbose=True):
        """
        Initializes a GUI viewer of the annotations.

        Attributes:
            _verbose: [bool] if True navigating actions will printout info
            _fr_sub_dirs: [list of str] where each str is an absolute path to the subdir for a camera
            _n_views: [int] number of views, equivalent to the subdirs found
            ann_filenames: [list of str] where each str is the absolute path of an annotation file
            current_frame: [int] ordering number of the annotation file currently shown
            corresponding_frames: [list of PIL.ImageTk.PhotoImage] corresponding frames, i.e. frames
                from different cameras (subdirs), with the same timestamp
            im_height: [int] height of each frame
            im_width: [int] width of each frame
            frames_ext: [str] extension of the frames, obtained from the input arguments (opt)

        GUI related attributes:
            canvas, frames_on_canvas, navigate_frame, & 4 buttons

        Raises:
            ValueError: if there is no subdirectory in the given opt.dir_frames,
                        or no annotation files in opt.dir_annotations
            NotADirectoryError: see funtions _subdirs & read_json


        :param _gui: [tkinter.Tk] window where the frames will be displayed
        :param _opt: [argparse.Namespace] required to initialize the Viewer
        :param _verbose: [bool] if True prints out info as the user navigates
        """
        self._verbose = _verbose
        self.fr_sub_dirs = _subdirs(opt.dir_frames)
        if len(self.fr_sub_dirs) == 0:
            raise ValueError('Could not find sub-directories in %s.' % opt.dir_frames)
        self.n_views = len(self.fr_sub_dirs)
        if self.n_views != _N_VIEWS:
            warnings.warn("Expected %d subdirectories, found %d." % (_N_VIEWS, len(self.fr_sub_dirs)))
        self.ann_filenames = _files(opt.dir_annotations, opt.ann_ext)
        if len(self.ann_filenames) == 0:
            raise ValueError('Could not find annotation files in %s with extension %s.'
                             % (opt.dir_annotations, opt.ann_ext))
        self.current_frame = 0
        self.corresponding_frames = [None for _ in range(self.n_views)]
        self.im_height = None
        self.im_width = None
        self.frames_ext = opt.fr_ext
        if self._verbose:
            print('Found %d files in: %s.' % (len(self.ann_filenames), _opt.dir_annotations))
            print('Loaded %d multi-view annotations.' %
                  sum([sum([1 for _ in read_json(f)]) for f in self.ann_filenames]))
        self.n_rows = 2
        self.n_columns = math.ceil((self.n_views + 1) / self.n_rows)
        self._load_and_draw_rect()

        # Set-up canvas
        self.canvas = Canvas(_gui,
                             width=((self.n_views + 1) // self.n_rows * self.im_width),
                             height=self.n_rows * self.im_height)
        self.frames_on_canvas = []
        for row in range(self.n_rows):
            for column in range(self.n_columns):
                if row*self.n_columns + column >= self.n_views:
                    break
                frame_on_canvas = Label(gui, image=self.corresponding_frames[row*self.n_columns+column])
                frame_on_canvas.image = self.corresponding_frames[row*self.n_columns+column]
                frame_on_canvas.grid(row=row, column=column)
                self.frames_on_canvas.append(frame_on_canvas)
        # frame for navigation
        self.navigate_frame = Canvas(_gui, width=self.im_width, height=self.im_height)
        self.navigate_frame.grid(row=self.n_rows-1, column=self.n_columns-1)
        # buttons for navigation
        self.prevBtn = Button(self.navigate_frame, text='<', command=lambda: self._on_button(-1))
        self.prevBtn.grid(row=0, column=0)
        self.nextBtn = Button(self.navigate_frame, text='>', command=lambda: self._on_button(1))
        self.nextBtn.grid(row=0, column=1)
        self.prev10Btn = Button(self.navigate_frame, text='-10 <<', command=lambda: self._on_button(-10))
        self.prev10Btn.grid(row=1, column=0)
        self.next10Btn = Button(self.navigate_frame, text='>> +10', command=lambda: self._on_button(10))
        self.next10Btn.grid(row=1, column=1)

    def _on_button(self, step):
        """
        Method called when button click occurs.
        It updates self.current_frame, calls method _load_and_draw_rect,
        and finally updates the displayed images with the newly loaded ones.
        Note that if bounds are reached (for e.g. user clicks previous, while
        the first frame is shown), the displayed images are not updated.
        
        :param step: [int] number of frames to subtract/add of the current one
        :return: [None]
        """
        _current_frame = self.current_frame + step

        if 0 <= _current_frame < len(self.ann_filenames):
            self.current_frame = _current_frame
            self._load_and_draw_rect()
            for view in range(self.n_views):
                frame = self.corresponding_frames[view]
                self.frames_on_canvas[view].configure(image=frame)

    def _load_and_draw_rect(self):
        """
        Loads the 'self.current_frame'-th frames and annotations.
        Loads the corresponding multi-view frames, 
        loads the annotations for these frames, 
        and finally draws the annotations on the frames.
        Updates self.corresponding_frames.
        Helper function used by self.__init__, and self._on_button.
        When called by __init__ it determines the size of the frames
        based on the width/height of the screen.
        
        Assumption: 
            The images are named as the corresponding annotation file.
            E.g. 000.json, and 000.jpeg in subdirectory view0, and view1.

        Raises:
            FileNotFoundError: if any of the frames for the given annotation file is not found.
                               See also function read_json.
            ValueError: See function read_json
            TypeError: See function read_json
        
        :return: [None]
        """
        _ann_filename = self.ann_filenames[self.current_frame]
        _frame_timestamp = _ann_filename[_ann_filename.rfind("/") + 1:_ann_filename.rfind(".")]
        _annotations = read_json(_ann_filename)

        for view in range(self.n_views):
            img_pth = self.fr_sub_dirs[view] + '/' + _frame_timestamp + self.frames_ext
            if not os.path.isfile(img_pth):
                raise FileNotFoundError("Corresponding frame %s of annotation file %s not found."
                                        % (img_pth, _ann_filename))
            frame = cv2.imread(img_pth)

            if self.im_height is None or self.im_width is None:
                # determine display size per image, s.t. the aspect ratio is maintained
                _im_height, _im_width, _ = frame.shape
                _screen_width = gui.winfo_screenwidth()
                _screen_height = gui.winfo_screenheight()
                _width_downscale_factor = _im_width * self.n_columns / (
                    _screen_width - (self.n_columns + 1) * 5)
                _height_downscale_factor = _im_height * self.n_rows / (
                    _screen_height - (self.n_rows + 1) * 5)
                _downscale_factor = max([_width_downscale_factor, _height_downscale_factor])
                self.im_height = int(_im_height / _downscale_factor)
                self.im_width = int(_im_width / _downscale_factor)

            for annotation in _annotations:
                bbox = annotation['views'][view]
                if self._visible(bbox) and self._validate_box(bbox):
                    cv2.rectangle(frame, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']),
                                  (255, 0, 0), 2)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = frame.resize((self.im_width, self.im_height), Image.ANTIALIAS)
            self.corresponding_frames[view] = ImageTk.PhotoImage(frame)

        if self._verbose:
            print("Frame %s [%3d/%3d]:\t%d multi-view annotations." %
                  (_frame_timestamp, self.current_frame + 1, len(self), len(_annotations)))

    @staticmethod
    def _visible(_box):
        """
        Checks if 3D position is visible for the particular view.
        A position is visible, iff each of the coordinates of the bounding
        box is different than -1.

        Raises:
            ValueError: if the input is not a dictionary

        :param _box: [dict] Bounding box coordinates, 
                    {'xmin': _, 'ymin': _, 'xmax': _, 'ymax': _},
                    where _ is an integer
        :return: [bool] True if visible, False otherwise
        """
        if not isinstance(_box, dict):
            raise ValueError(f"Type mismatch. Found {type(_box)}, expected dict.")
        return -1 not in _box.values()

    @staticmethod
    def _validate_box(_box):
        """
        Checks if a bounding box (BB) is valid.
        Given BB is valid if xmin < xmax & ymin < ymax.

        Raises:
            ValueError: if the input is not a dictionary

        :param _box: [dict] Bounding box coordinates, {'xmin': _, 'ymin': _, 'xmax': _, 'ymax': _}
        :return: [bool] True if valid, False otherwise
        """
        if not isinstance(_box, dict):
            raise ValueError(f"Type mismatch. Found {type(_box)}, expected dict.")
        return True if _box['xmin'] < _box['xmax'] and _box['ymin'] < _box['ymax'] else False

    def __len__(self):
        return len(self.ann_filenames)


def parse_args():
    """
    Parses the input command line arguments, and checks if the given
    directories of the frames and the annotations exist.

    :return: [argparse.Namespace]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_annotations', default='annotations',
                        help='directory where the annotations are stored, default: %(default)s')
    parser.add_argument('--dir_frames', default='frames',
                        help='root directory which contains the frames in separate subdirectories per view, '
                             'default: %(default)s', )
    parser.add_argument('--ann_ext', default='.json', help='extension of the annotations, default: %(default)s')
    parser.add_argument('--fr_ext', type=str, default='.png',
                        help='extension of the frames, default: %(default)s')
    _opt = parser.parse_args()
    assert os.path.exists(_opt.dir_annotations), "Directory not found: %s." % _opt.dir_annotations
    assert os.path.exists(_opt.dir_frames), "Directory not found: %s." % _opt.dir_frames
    return _opt


if __name__ == '__main__':
    opt = parse_args()
    gui = init_window("WILDTRACK dataset")
    AnnotationsViewer(_gui=gui, _opt=opt)
    close_window(gui)


