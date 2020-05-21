"""
Ffmpeg-based video file reader with timestamp support and optional GPU decoding
"""
import os
import re
import time
from typing import Union, Tuple

import cv2
import numpy as np
import subprocess
import threading
import logging

logger = logging.getLogger()


class FfmpegCapture:
	# how many times to poll for timestamp availability before generating error
	MAX_TIMESTAMP_WAIT = 100
	TIMESTAMP_POLL_INTERVAL = 0.01

	def __init__(self, filename: str, use_gpu=False):
		if not os.path.exists(filename):
			raise ValueError(f'File {filename} doesn\'t exist')
		self.stream_thread: threading.Thread
		self.filename = filename
		self.started = False
		self.stopping = False
		self.timestamps = []
		self.frame_idx = 0
		self.decoder = ''
		if use_gpu:
			self.decoder = '-hwaccel cuvid -c:v h264_cuvid'
		self._read_metadata()

	def _read_metadata(self):
		"""
		Reads video properties and fills corresponding fields
		@return:
		"""
		cap = None
		try:
			cap = cv2.VideoCapture(self.filename)
			self.fps = cap.get(cv2.CAP_PROP_FPS)
			self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			if not self.width or not self.height:
				# have to read frame to get dimensions
				frame = cap.read()
				self.height, self.width = frame.shape[:2]
			logger.info(f'Video file opened {self.filename}, {self.width}x{self.height}, {self.fps} FPS')
		finally:
			if cap is not None:
				cap.release()

	def read(self) -> Union[Tuple[int, float, np.ndarray], Tuple[None, None, None]]:
		"""
		Reads next frame from video.
		@return: Tuple[frame_index, frame_timestamp, frame] or [None, None, None] if end of video
		"""
		if not self.started:
			self.start()
		# get raw frame from stdout and convert it to numpy array
		bytes = self.process.stdout.read(self.height * self.width * 3)
		if len(bytes) == 0:
			return None, None, None
		frame = np.frombuffer(bytes, np.uint8).reshape([self.height, self.width, 3])
		timestamp = self._get_timestamp_for_frame(self.frame_idx)
		logger.debug(f'Read frame {self.frame_idx} at PTS_TIME {timestamp}')
		self.frame_idx += 1
		return self.frame_idx, timestamp, frame

	def _get_timestamp_for_frame(self, frame_idx) -> float:
		# wait for timestamp record to be available, normally it available before frame is read
		waits = 0
		while frame_idx > len(self.timestamps) - 1:
			time.sleep(FfmpegCapture.TIMESTAMP_POLL_INTERVAL)
			waits += 1
			if waits > FfmpegCapture.MAX_TIMESTAMP_WAIT:
				raise Exception('Error reading video timestamps')
		if waits > 0:
			logger.debug(f'Waited for frame timestamp for {FfmpegCapture.TIMESTAMP_POLL_INTERVAL * waits} sec')
		return self.timestamps[frame_idx]

	def start(self):
		# start ffmpeg process
		ffmpeg_cmd = f"ffmpeg -y -debug_ts -hide_banner -i {self.decoder} {self.filename} -copyts -f rawvideo -pix_fmt bgr24 pipe:"
		self.process = subprocess.Popen(ffmpeg_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		# stderr and stdout are not synchronized, read timestamp data in separate thread
		self.stream_thread = threading.Thread(target=self.stream_reader, args=[self.process.stderr])
		self.stream_thread.start()
		# wait for stream reader thread to fill timestamp list
		time.sleep(0.05)
		self.started = True

	def stream_reader(self, stream):
		while not self.stopping:
			try:
				last_line = stream.readline().decode('ascii')
				if not last_line:
					break
				m = re.match('^demuxer\+ffmpeg -> ist_index:[0-9].+type:video.+pkt_pts_time:(?P<pkt_pts_time>\d*\.?\d*)', last_line)
				if m:
					self.timestamps.append(float(m.group('pkt_pts_time')))
			except:
				if not self.stopping:
					raise

	def release(self):
		"""
		Stop Ffmpeg instance
		@return:
		"""
		try:
			if self.started:
				self.stopping = True
				self.process.terminate()
		except:
			pass