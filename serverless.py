from itertools import tee
from dotenv import load_dotenv
import runpod
import time
import boto3
import os
import io
import sys  # Added to process command-line arguments
import logging
import warnings

from src.translation_pipeline import VideoTranslationPipeline
from src.translation_pipeline import ChangeManager
from src.pipeline_models.models import TranslationPipelineConfig
from src.pipeline_models.models import TraslationUpdate
from src.file_repository import RemoteFile, RemoteFileRepository, LocalFileRepository

load_dotenv()

# Set the global logging level to WARNING to hide INFO messages
logging.disable(logging.DEBUG)


def init_s3_client():
    s3 = boto3.client('s3',
        endpoint_url            = 'https://storage.yandexcloud.net',
        aws_access_key_id       = os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key   = os.environ['AWS_SECRET_ACCESS_KEY']
    )
    return s3

BASE_DIR = "data"

def create_videotranslate_config(source_lang, 
                                 target_lang, 
                                 name, 
                                 public_id, 
                                 num_speakers, 
                                 tts_engine, 
                                 file_path, 
                                 token,
                                 watermark
    ):
    config = TranslationPipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        name=name,
        public_id=public_id,
        num_speakers=num_speakers,
        source_video_path=file_path,
        base_dir=BASE_DIR,
        device="cuda",
        voice_conv=False,
        tts_model=tts_engine,
        dubbing_algo="speedup",
        eleven_api_token=token,
        watermark=watermark
    ) 
    return config

def get_file(repo, s3_url):
    remote_file = RemoteFile(
            s3_url = s3_url,
            name="source.mp4"
    )
    remote_file = repo.materialize_file(remote_file)
    return remote_file.file_path

def handler(job):
    input = job['input']
    
    source_language = input.get('source_language', None)
    target_language = input.get('target_language', "english")
    tts_engine = input.get("tts_engine", "xtts")
    token = input.get("token", None)
            
    num_speakers = input.get('count_speakers', None)
    name = input.get('name', "video_translation")
    public_id = input.get('public_id', "public_id")
    s3_video_url = input.get("s3_video_url")
    watermark = input.get("watermark", True)

    # First progress update - Initialization
    runpod.serverless.progress_update(job, "0% Initializing translation pipeline")

    s3_client = init_s3_client()
    repo = RemoteFileRepository(public_id, BASE_DIR, s3_client)
    file_path = get_file(repo, s3_video_url)
    
    config = create_videotranslate_config(
        source_language, 
        target_language, 
        name, 
        public_id, 
        num_speakers, 
        tts_engine, 
        file_path, 
        token,
        watermark
    )
    
    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    
    # Create a wrapped version of the pipeline that adds progress updates
    # Second progress update - Transcription
    runpod.serverless.progress_update(job, "20% Starting transcription (Speech-to-Text)")
    pipeline._generate_asr()
    
    # Third progress update - Translation
    runpod.serverless.progress_update(job, "40% Starting translation")
    pipeline._generate_translation()
    
    # Fourth progress update - Text-to-Speech
    runpod.serverless.progress_update(job, "60% Starting Text-to-Speech synthesis")
    pipeline._generate_speech()
    
    # Fifth progress update - Audio separation and merging
    runpod.serverless.progress_update(job, "80% Starting audio separation and enhancement")
    video_translation = pipeline._merge(pipeline.config.dubbing_algo)
    
    # Final progress update
    runpod.serverless.progress_update(job, "100% Translation pipeline completed successfully")
    
    result_video = video_translation.processed_video
    
    return {'s3_result_video_url': f'{result_video.s3_url}'}

def test_video_translation_local():
    """
    This function builds a sample event and calls the handler.
    Adjust the values for testing based on your environment.
    """
    test_event = {
        "input": {
          #  "source_language": "russian",          # change as needed
            "target_language": "spanish",           # change as needed
            "tts_engine": "f5tts",                   # change as needed
            "token": "your_eleven_api_token",       # provide a valid token if required
      #      "count_speakers": 1,
            "name": "local_test_video_translation",
            "public_id": "local_test_id",
            "s3_video_url": "https://url_on_your_video_on_s3",  # provide a valid S3 URL
            "watermark": True   
        }
    }
    try:
        result = handler(test_event)
        print("Translation Test Successful:")
        print(result)
    except Exception as e:
        print("Translation Test Failed:")
        print(e)

if __name__ == '__main__':
    # If a command-line argument 'test' is given, run the test function
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running local translation test...")
        test_video_translation_local()
    else:
        runpod.serverless.start({'handler': handler})