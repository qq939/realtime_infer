#!/usr/bin/env python3
import unittest
import sys
import os
import time
import threading
import tempfile
import cv2
import numpy as np
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

class TestPyQt5VideoAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_dir = Path("src")
        cls.dst_dir = Path("dst")
        cls.src_dir.mkdir(exist_ok=True)
        cls.dst_dir.mkdir(exist_ok=True)
    
    def test_00_pyqt5_imports(self):
        from PyQt5 import QtWidgets, QtCore, QtGui
        self.assertIsNotNone(QtWidgets)
        self.assertIsNotNone(QtCore)
        self.assertIsNotNone(QtGui)
    
    def test_01_video_file_exists(self):
        video_files = list(self.src_dir.glob("*.mp4"))
        self.assertGreater(len(video_files), 0, "src目录下应该有视频文件")
    
    def test_02_pyqt5_window_creation(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        self.assertIsNotNone(annotator)
        self.assertEqual(annotator.video_path, str(video_files[0]))
    
    def test_03_draw_masks_function(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        
        frame = annotator.frame
        self.assertIsNotNone(frame)
        fh, fw = frame.shape[:2]
        
        annotator.current_masks = []
        annotator.added_points = []
        annotator.deleted_mask_ids = {}
        
        result = annotator.draw_masks_on_frame(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)
    
    def test_04_draw_masks_with_single_mask(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        
        frame = annotator.frame
        fh, fw = frame.shape[:2]
        
        mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.rectangle(mask, (fw//4, fh//4), (3*fw//4, 3*fh//4), 255, -1)
        
        annotator.current_masks = [(0, mask)]
        annotator.added_points = []
        annotator.deleted_mask_ids = {}
        
        result = annotator.draw_masks_on_frame(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)
    
    def test_05_draw_masks_with_resized_mask(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        
        frame = annotator.frame
        fh, fw = frame.shape[:2]
        
        mask_small = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask_small, (10, 10), (90, 90), 255, -1)
        
        annotator.current_masks = [(0, mask_small)]
        annotator.added_points = []
        annotator.deleted_mask_ids = {}
        
        result = annotator.draw_masks_on_frame(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)
    
    def test_06_draw_masks_with_added_points(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        
        frame = annotator.frame
        annotator.current_masks = []
        annotator.added_points = [[100, 100], [200, 200]]
        annotator.deleted_mask_ids = {}
        
        result = annotator.draw_masks_on_frame(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)
    
    def test_07_draw_masks_with_deleted_masks(self):
        from PyQt5.QtWidgets import QApplication
        from annotate_video import PyQt5VideoAnnotator
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        annotator = PyQt5VideoAnnotator(str(video_files[0]), str(self.dst_dir))
        
        frame = annotator.frame
        fh, fw = frame.shape[:2]
        
        mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.rectangle(mask, (fw//4, fh//4), (3*fw//4, 3*fh//4), 255, -1)
        
        annotator.current_masks = [(0, mask)]
        annotator.added_points = []
        annotator.deleted_mask_ids = {0: mask}
        
        result = annotator.draw_masks_on_frame(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)
    
    def test_08_video_capture(self):
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")
        
        cap = cv2.VideoCapture(str(video_files[0]))
        self.assertTrue(cap.isOpened())
        
        ret, frame = cap.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        
        fh, fw = frame.shape[:2]
        self.assertGreater(fh, 0)
        self.assertGreater(fw, 0)
        
        cap.release()
    
    def test_09_global_parameters(self):
        from annotate_video import FIND, SRC_DIR, DST_DIR, TEMP_DATA_DIR, WINDOW_NAME
        self.assertIsNotNone(FIND)
        self.assertIsNotNone(SRC_DIR)
        self.assertIsNotNone(DST_DIR)
        self.assertIsNotNone(WINDOW_NAME)

if __name__ == '__main__':
    print(f"Python: {sys.version}")
    print(f"Test start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    unittest.main(verbosity=2)