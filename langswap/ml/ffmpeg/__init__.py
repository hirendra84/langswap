import shlex
import subprocess
from enum import Enum, auto
import os
from tempfile import NamedTemporaryFile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Util(Enum):
    ffmpeg = auto()
    ffprobe = auto()

class FFmpegClient:
    def __init__(self, ffmpeg_path='ffmpeg', ffprobe_path='ffprobe'):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def run_command(self, command, util: Util = Util.ffmpeg):
        """Run a custom FFmpeg command quietly by ensuring the quiet log level is set."""
        command = f"-hide_banner -loglevel error {command}"
        
        util_path = self.ffprobe_path if util == Util.ffprobe else self.ffmpeg_path
        process = subprocess.run(shlex.split(f"{util_path} {command}"),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        if process.stderr:
            logger.debug(f"Command error: {process.stderr}")
        return process.stdout, process.stderr

    def convert_video(self, input_path, output_path, output_format):
        """Convert video to different formats."""
        cmd = f"-y -i {input_path} -c:v libx264 -crf 23 -preset fast {output_path}.{output_format}"
        return self.run_command(cmd)

    def extract_audio(self, input_path, output_path, time_limit: int | None = None, target_sr=24000):
        """Extract audio from video."""
        cmd = f"-y -i {input_path} -vn -acodec pcm_s16le -ar {target_sr} -ac 1 -f wav {output_path}"
        return self.run_command(cmd)

    def resample_audio(self, input_path, output_path, sample_rate: int = 16_000):
        cmd = f"-y -i {input_path} -ar {sample_rate} -f wav {output_path}"
        return self.run_command(cmd)

    def get_audio_length(self, input_path) -> float:
        cmd = f'-i {input_path} -show_entries format=duration -v quiet -of csv="p=0"'
        res, err = self.run_command(cmd, Util.ffprobe)
        if err:
            raise ValueError(err)
        return float(res)

    def replace_audio(self, video_input_path: str, audio_input_path: str,
                      video_output_path: str,
                      time_limit: int | None = None):
        limit_command = ''
        if time_limit:
            limit_command = f'-t {time_limit}'
        logger.debug(f"Running command: {video_input_path} {audio_input_path} {video_output_path} {limit_command}")
        cmd = (f'-y -i {video_input_path} -i {audio_input_path} -c:v '
               f'copy -map 0:v:0 -map 1:a:0 {limit_command} '
               f'-f mp4 -shortest {video_output_path}')
        return self.run_command(cmd)

    def add_watermark(self, input_path: str, output_path: str,
                      text: str = "translated with langswap.app",
                      fontcolor: str = "white",
                      fontsize: int = 16,
                      x: str = "w-tw-10",
                      y: str = "h-th-10",
                      fontfile: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        """Adds a watermark using FFmpeg's drawtext filter. Handles in-place operations via a temporary file."""
        in_place = os.path.abspath(input_path) == os.path.abspath(output_path)
        if in_place:
            tmp_dir, ext = os.path.dirname(input_path), os.path.splitext(input_path)[1]
            with NamedTemporaryFile(suffix=ext, dir=tmp_dir, delete=False) as tmp:
                temp_output_path = tmp.name
        else:
            temp_output_path = output_path

        cmd = (f'-y -i {input_path} '
               f'-vf "drawtext=text=\'{text}\':fontfile={fontfile}:fontcolor={fontcolor}:'
               f'fontsize={fontsize}:x={x}:y={y}" -codec:a copy {temp_output_path}')
        stdout, stderr = self.run_command(cmd)
        # logger.info(f"stdout: {stdout}\nstderr: {stderr}")
        if not os.path.exists(temp_output_path):
            raise ValueError(f"FFmpeg failed to produce an output file. Error: {stderr}")
        if in_place:
            os.replace(temp_output_path, input_path)
        return stdout, stderr
