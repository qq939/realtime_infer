import unittest
import sys
import os
import time
import threading
from pathlib import Path

class TestPyQt5AnnotateVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src_dir = Path("src")
        cls.dst_dir = Path("dst")
        cls.src_dir.mkdir(exist_ok=True)
        cls.dst_dir.mkdir(exist_ok=True)

    def test_pyqt5_import(self):
        from PyQt5 import QtWidgets
        self.assertIsNotNone(QtWidgets)
    
    def test_pyqt5_qtcore_import(self):
        from PyQt5 import QtCore
        self.assertIsNotNone(QtCore)
    
    def test_pyqt5_qtgui_import(self):
        from PyQt5 import QtGui
        self.assertIsNotNone(QtGui)

    def test_qapplication_creation(self):
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        self.assertIsNotNone(app)

    def test_video_file_exists(self):
        video_files = list(self.src_dir.glob("*.mp4"))
        self.assertGreater(len(video_files), 0, "src目录下应该有视频文件")

    def test_video_capture_opens(self):
        import cv2
        video_files = list(self.src_dir.glob("*.mp4"))
        if len(video_files) == 0:
            self.skipTest("没有测试视频文件")

        cap = cv2.VideoCapture(str(video_files[0]))
        self.assertTrue(cap.isOpened(), "视频应该能正常打开")
        ret, frame = cap.read()
        self.assertTrue(ret, "应该能读取视频帧")
        cap.release()

    def test_annotation_data_structure(self):
        from annotate_video import AnnotationBox
        box = AnnotationBox(100, 100, 200, 200, (0, 255, 0))
        self.assertEqual(box.x1, 100)
        self.assertEqual(box.y1, 100)
        self.assertEqual(box.x2, 200)
        self.assertEqual(box.y2, 200)
        self.assertEqual(box.color, (0, 255, 0))

    def test_track_manager_exists(self):
        from annotate_video import TrackManager
        tm = TrackManager(iou_threshold=0.5)
        self.assertIsNotNone(tm)
        self.assertEqual(tm.iou_threshold, 0.5)

    def test_box_colors_defined(self):
        from annotate_video import BOX_COLORS
        self.assertGreater(len(BOX_COLORS), 0, " BOX_COLORS 应该已定义")
        self.assertEqual(len(BOX_COLORS[0]), 3, "每个颜色应该是RGB三元组")

    def test_timeout_mechanism(self):
        timeout_seconds = 5
        start_time = time.time()

        def long_running_task():
            time.sleep(10)

        thread = threading.Thread(target=long_running_task)
        thread.start()
        thread.join(timeout=timeout_seconds)

        elapsed = time.time() - start_time
        self.assertLessEqual(elapsed, timeout_seconds + 1, "任务应该在超时时间内完成")
        if thread.is_alive():
            thread.join(1)

    def test_global_parameters_defined(self):
        from annotate_video import FIND, SRC_DIR, DST_DIR, TEMP_DATA_DIR, WINDOW_NAME
        self.assertIsNotNone(FIND)
        self.assertIsNotNone(SRC_DIR)
        self.assertIsNotNone(DST_DIR)
        self.assertIsNotNone(TEMP_DATA_DIR)
        self.assertIsNotNone(WINDOW_NAME)

    def test_helper_functions_exist(self):
        from annotate_video import put_chinese_text, calculate_bbox_iou, calculate_mask_iou
        self.assertTrue(callable(put_chinese_text))
        self.assertTrue(callable(calculate_bbox_iou))
        self.assertTrue(callable(calculate_mask_iou))

if __name__ == '__main__':
    unittest.main(verbosity=2)