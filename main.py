import uuid
from dotenv import load_dotenv
import boto3
import os
import logging

from src.translation_pipeline import VideoTranslationPipeline
from src.pipeline_models.models import TranslationPipelineConfig
from src.file_repository import RemoteFile, RemoteFileRepository

load_dotenv()

# Set the global logging level to WARNING to hide INFO messages
logging.disable(logging.DEBUG)

BASE_DIR = "data"

def init_s3_client():
    s3 = boto3.client('s3',
        endpoint_url            = 'https://storage.yandexcloud.net',
        aws_access_key_id       = os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key   = os.environ['AWS_SECRET_ACCESS_KEY']
    )
    return s3

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

def process_translation(input, progress_callback=None):
    """
    Core translation processing function that can be used both by RunPod handler
    and local testing
    
    Args:
        input: Dictionary with translation parameters
        progress_callback: Optional function to report progress
    """
    # Helper function for progress updates
    def update_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
    
    source_language = input.get('source_language', None)
    target_language = input.get('target_language', "english")
    tts_engine = input.get("tts_engine", "xtts") # xtts, f5tts, elevenlabs
    token = input.get("token", None)
            
    num_speakers = input.get('count_speakers', None)
    random_id = str(uuid.uuid4())
    name = input.get('name', random_id)
    public_id = input.get('public_id', random_id)
    s3_video_url = input.get("s3_video_url")
    watermark = input.get("watermark", True)
    
    # First progress update - Initialization
    update_progress("0% Initializing translation pipeline")

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
    
    # Second progress update - Transcription
    update_progress("20% Starting transcription (Speech-to-Text)")
    pipeline._generate_asr()
    
    # Third progress update - Translation
    update_progress("40% Starting translation")
    pipeline._generate_translation()
    
    # Fourth progress update - Text-to-Speech
    update_progress("60% Starting Text-to-Speech synthesis")
    pipeline._generate_speech()
    
    # Fifth progress update - Audio separation and merging
    update_progress("80% Starting audio separation and enhancement")
    video_translation = pipeline._merge(pipeline.config.dubbing_algo)
    
    # Generate SRT files - updated to use the method from the pipeline
    update_progress("95% Generating subtitle files")
    source_srt, translated_srt = pipeline.generate_srt_files()
    
    # Final progress update
    update_progress("100% Translation pipeline completed successfully")
    
    result_video = video_translation.processed_video
    
    # Return URLs for both the video and the SRT files
    return {
        's3_result_video_url': f'{result_video.s3_url}',
        's3_source_transcript_url': f'{source_srt.s3_url}',
        's3_translated_transcript_url': f'{translated_srt.s3_url}'
    }

def test_video_translation_local(input_file="test_input.json"):
    """
    This function builds a sample event and calls the handler.
    Adjust the values for testing based on your environment.
    """
    import json
    with open(input_file, "r") as f:
        test_event = json.load(f)
    try:
        result = process_translation(test_event['input'])
        print("Translation Test Successful:")
        print(result)
        return result
    except Exception as e:
        print("Translation Test Failed:")
        print(e)
        raise

if __name__ == "__main__":
    test_video_translation_local()