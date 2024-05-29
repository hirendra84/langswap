from src.ml.api_client import MockAPIClient
from src.file_repository import LocalFileRepository
from src.pipeline_models.models import RemoteFile
from src.pipeline_models.models import VideoTranslation
from src.settings import BASE_WORKING_DIR
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.s3_client import get_s3_client

api_client_klass = MockAPIClient

api_client = api_client_klass('dontcare')


def main():

    public_id = 'some_random_public_id'

    file_repository = LocalFileRepository(
            public_id,
            base_directory=BASE_WORKING_DIR,
            s3_client=get_s3_client()
        )

    file = RemoteFile(
        name='some_name',
        s3_url='https://ds-dev-video-storage.s3.amazonaws.com//Users/nikolaypakhtusov/data/10d64ca2-19ca-11ef-b490-9a1744b66515/extracted_audio_resampled_16000_vad.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=***REDACTED-AWS-KEY-ID***%2F20240524%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20240524T123525Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=5a3e384aea983f83d5d2364df4ba87be8aa540438ecd1c7f5f6dbf2f4ad1e374',
    )

    manager = SpeechToTextManager(public_id, api_client, file_repository)
    video_translation = manager.extract_and_transcribe(VideoTranslation(source_file=file))
    manager = TranslationManager(public_id, api_client)
    video_translation = manager.translate(video_translation)
    manager = TextToSpeechManager(public_id, api_client, file_repository)
    video_translation = manager.synthesize(video_translation)
    print(video_translation)


if __name__ == '__main__':
    main()
