import argparse
import os
import torch
from src.ml.api_client import MockAPIClient
from src.file_repository import LocalFileRepository
from src.pipeline_models.models import RemoteFile, VideoTranslation
from src.utils.s3_client import get_s3_client
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.logging import Logger

def main(args):
    print("local check")
    api_client = MockAPIClient('dontcare')
    
    file_repository = LocalFileRepository(
        args.public_id,
        base_directory=args.base_dir,
        s3_client=get_s3_client()
    )

    file = RemoteFile(
        file_path=args.file_path,
        name=args.name
    )
    file = file_repository.save_file(file, force=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    logger = Logger(directory=file_repository.directory)

    video_translation = VideoTranslation(source_file=file, public_id=args.public_id)

    manager = SpeechToTextManager(args.public_id, api_client, file_repository, device="cuda", logger=logger)
    video_translation = manager.extract_and_transcribe(video_translation, lang=args.source_lang)

    torch.cuda.empty_cache()
    manager = TranslationManager(args.public_id, api_client, file_repository, device="cuda:1", logger=logger)
    video_translation = manager.translate(video_translation, source_lang=args.source_lang, target_lang=args.target_lang)
    torch.cuda.empty_cache()

    manager = TextToSpeechManager(args.public_id, api_client, file_repository, tts_sample_rate=24000, device="cuda:1", logger=logger)
    video_translation = manager.synthesize(video_translation, source_lang=args.source_lang, target_lang=args.target_lang, voice_conv=False, merge_pipeline="stretch_whole", enhance=True)
    print(video_translation)

if __name__ == "__main__":
    # python pipeline_check.py --file_path /path/to/your/video.mp4 --base_dir /path/to/base/directory --source_lang russian --target_lang english --name your_video_name
    parser = argparse.ArgumentParser(description="Video translation pipeline")
    parser.add_argument("--file_path", required=True, help="Path to the input video file")
    parser.add_argument("--base_dir", required=True, help="Base directory for file repository")
    parser.add_argument("--source_lang", required=True, help="Source language")
    parser.add_argument("--target_lang", required=True, help="Target language")
    parser.add_argument("--name", required=True, help="Name of the video file")
    parser.add_argument("--public_id", default="some_random_public_id", help="Public ID for the video")

    args = parser.parse_args()
    main(args)


