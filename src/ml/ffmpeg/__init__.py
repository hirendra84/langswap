import shlex
import subprocess
from enum import Enum, auto


class Util(Enum):
    ffmpeg = auto()
    ffprobe = auto()

class FFmpegClient:
    def __init__(self, ffmpeg_path='ffmpeg', ffprobe_path='ffprobe'):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def run_command(self, command, util: Util = Util.ffmpeg):
        """Run a custom FFmpeg command."""
        util_path = self.ffprobe_path if util == Util.ffprobe else self.ffmpeg_path

        process = subprocess.run(shlex.split(f"{util_path} {command}"),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        return process.stdout, process.stderr

    def convert_video(self, input_path, output_path, output_format):
        """Convert video to different formats."""
        cmd = f"-y -i {input_path} -c:v libx264 -crf 23 -preset fast {output_path}.{output_format}"
        return self.run_command(cmd)

    def extract_audio(self, input_path, output_path, time_limit: int | None = None, target_sr=24000):
        """Extract audio from video."""
        cmd = f"-y -i {input_path} -vn -map 0:a:0 -af aresample=resampler=soxr -ar {target_sr} -ac 1 -fflags +shortest -max_interleave_delta 0 -f wav {output_path}"

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

        cmd = (f'-i {video_input_path} -i {audio_input_path} -c:v '
               f'copy -map 0:v:0 -map 1:a:0 {limit_command}'
               f'-f mp4 -shortest {video_output_path}')
        return self.run_command(cmd)
