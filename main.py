import uuid
from dotenv import load_dotenv
import boto3
import os
import logging

from langswap.translation_pipeline import VideoTranslationPipeline, ChangeManager
from langswap.pipeline_models.models import TraslationUpdate
from langswap.pipeline_models.models import TranslationPipelineConfig, load_config_from_json
from langswap.file_repository import RemoteFile, RemoteFileRepository, download_s3_directory

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

    assert 'target_language' in input.keys(), "target language is missing from input"
    assert 'tts_engine' in input.keys(), "tts engine is missing from input"
    
    # First progress update - Initialization
    update_progress("0% Initializing translation pipeline")

    s3_client = init_s3_client()

    public_id = input.get('public_id', str(uuid.uuid4()))
    repo = RemoteFileRepository(public_id, BASE_DIR, s3_client)
    file_path = get_file(repo, input.get("s3_video_url"))
    
    tts_engine = input.get("tts_engine", "chatterbox")
    if input.get('source_language') == "english" and input.get("target_language") == "russian" and input.get("tts_engine") == "xtts":
        tts_engine = "f5tts"

    config = TranslationPipelineConfig(
        source_lang=input.get('source_language', None),
        target_lang=input.get('target_language'),
        name=public_id,
        public_id=public_id,
        num_speakers=input.get('count_speakers', None),
        source_video_path=file_path,
        base_dir=BASE_DIR,
        device='cuda',
        voice_conv=False,
        tts_model=tts_engine,
        dubbing_algo=input.get("dubbing_algo", "speedup"),
        eleven_api_token=input.get("token", None),
        watermark=input.get("watermark", True),
        asr_backend=input.get("asr_backend", "qwen"),
        translation_backend=input.get("translation_backend", "local"),
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

def process_update_translation(input, progress_callback=None):
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
    
    public_id = input.get('public_id')
    s3_video_url = input.get("s3_video_url")
    update_translation_collection = input.get("update_translation")
    
    # First progress update - Initialization
    update_progress("0% Initializing translation pipeline")

    s3_client = init_s3_client()
    repo = RemoteFileRepository(public_id, BASE_DIR, s3_client)
    get_file(repo, s3_video_url)
    bucket = os.getenv('BUCKET')
    download_s3_directory(s3_client, bucket, f"{BASE_DIR}/{public_id}", f"{BASE_DIR}/{public_id}")
    config_file = repo.get_file("config.json")
    config = load_config_from_json(config_file.file_path)
    
    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    # Second progress update - Transcription
    update_progress("30% Loading cache")
    video_translation = pipeline.translate_video()

  
    update_progress("60% Initializing change manager")
    change_menager = ChangeManager(pipeline, video_translation)
    
    video_translation = change_menager.apply_update_translations(TraslationUpdate.from_pairs(update_translation_collection))
    update_progress("90% Apply changes")
    update_progress("95% Generating subtitle files")
    pipeline.video_translation = video_translation
    source_srt, translated_srt = pipeline.generate_srt_files()
    
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

def test_local_file(
    local_video_path: str,
    source_language: str = "russian",
    target_language: str = "english",
    tts_engine: str = "omnivoice",
    device: str = "mps",
    skip_diarization: bool = True,
    asr_backend: str = "qwen",
    translation_backend: str = "local",
):
    """
    Run the full pipeline on a local video file without S3.
    Useful for local dev/testing on Mac.
    """
    from langswap.file_repository import LocalOnlyFileRepository
    from langswap.translation_pipeline import VideoTranslationPipeline
    from langswap.pipeline_models.models import TranslationPipelineConfig

    # Derive a stable ID from the input file so intermediate results are cached
    # across re-runs (avoids re-running ASR/translation on every retry).
    import hashlib
    public_id = hashlib.md5(str(local_video_path).encode()).hexdigest()[:12]
    repo = LocalOnlyFileRepository(public_id, BASE_DIR)

    config = TranslationPipelineConfig(
        source_lang=source_language,
        target_lang=target_language,
        name=public_id,
        public_id=public_id,
        num_speakers=None,
        source_video_path=local_video_path,
        base_dir=BASE_DIR,
        device=device,
        voice_conv=False,
        tts_model=tts_engine,
        dubbing_algo="speedup",
        eleven_api_token=os.environ.get("ELEVEN_API_KEY"),
        watermark=False,
        skip_diarization=skip_diarization,
        asr_backend=asr_backend,
        translation_backend=translation_backend,
    )

    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)

    print("20% Starting transcription (Speech-to-Text)")
    pipeline._generate_asr()
    print("40% Starting translation")
    pipeline._generate_translation()
    print("60% Starting Text-to-Speech synthesis")
    pipeline._generate_speech()
    print("80% Starting audio separation and enhancement")
    video_translation = pipeline._merge(pipeline.config.dubbing_algo)
    print("95% Generating subtitle files")
    source_srt, translated_srt = pipeline.generate_srt_files()
    print("100% Done")
    print("Output:", video_translation.processed_video.file_path)
    return video_translation


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        # Usage: python main.py local <video> [asr] [translation] [tts] [src_lang] [tgt_lang]
        # Examples:
        #   python main.py local video.mp4
        #   python main.py local video.mp4 openai openai omnivoice english russian
        video = sys.argv[2] if len(sys.argv) > 2 else "test_videos/tanks.mp4"
        asr_b = sys.argv[3] if len(sys.argv) > 3 else "qwen"
        tr_b  = sys.argv[4] if len(sys.argv) > 4 else "local"
        tts_e = sys.argv[5] if len(sys.argv) > 5 else "omnivoice"
        src_l = sys.argv[6] if len(sys.argv) > 6 else "russian"
        tgt_l = sys.argv[7] if len(sys.argv) > 7 else "english"
        test_local_file(video, source_language=src_l, target_language=tgt_l,
                        asr_backend=asr_b, translation_backend=tr_b, tts_engine=tts_e)
    else:
        test_video_translation_local()