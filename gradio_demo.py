import gradio as gr
import os
import torch
from pathlib import Path
from src.ml.api_client import MockAPIClient
from src.file_repository import LocalFileRepository
from src.pipeline_models.models import RemoteFile, VideoTranslation
from src.utils.s3_client import get_s3_client
from src.ml.speech_to_text_service import SpeechToTextManager
from src.ml.text_to_speech_service import TextToSpeechManager
from src.ml.translation_service import TranslationManager
from src.utils.logging import Logger
import shutil
from gradio_log import Log

base_dir = "/app/data"

log_file_path = "/app/data/folder/logs.log"
# Supported languages
LANGUAGES_SOURCE = [
    "Arabic", "Chinese (simplified)", "Chinese (traditional)", "Czech", "Dutch", "English", "French", "German",
    "Greek", "Hebrew", "Hindi", "Indonesian", "Italian", "Japanese", "Korean", "Persian", "Polish",
    "Portuguese", "Romanian", "Russian", "Spanish", "Turkish", "Ukrainian", "Vietnamese"
]
LANGUAGES_TARGET =[
    "English", "Russian",
]
TTS_MODELS = ["f5tts", "xtts", "elevenlabs"]
DUBBING_ALGO = ["speedup", "pause_based", "stretch_whole"]

def video_translation_pipeline(file_path_upload_video, source_lang, target_lang, tts_model, dubbing_algo, eleven_api_token, num_speakers):

    source_lang = source_lang.lower()
    target_lang = target_lang.lower()
    num_speakers = int(num_speakers)
    public_id = "folder"

    dir_of_video = f"{base_dir}/{public_id}/"
    if os.path.exists(dir_of_video):
        for item in os.listdir(dir_of_video):
            item_path = os.path.join(dir_of_video, item)
            
            if item == "logs.log":
                continue
            
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  
                print(f"Removed directory: {item_path}")
            else:
                os.remove(item_path)
                print(f"Removed file: {item_path}")
    else:
        print(f"File does not exist: {dir_of_video}")

    api_client = MockAPIClient('dontcare')
    
    file_repository = LocalFileRepository(
        public_id,
        base_directory=base_dir,
        s3_client=get_s3_client()
    )
    file = RemoteFile(
        file_path=file_path_upload_video,
        name="name"
    )
    file = file_repository.save_file(file, force=False)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    logger = Logger(directory=file_repository.directory)

    video_translation = VideoTranslation(source_file=file, public_id=public_id)

    manager = SpeechToTextManager(public_id, api_client, file_repository, device="cuda", logger=logger)
    video_translation = manager.extract_and_transcribe(video_translation, num_speakers=num_speakers, lang=source_lang)

    torch.cuda.empty_cache()
    manager = TranslationManager(public_id, api_client, file_repository, device="cuda:1", logger=logger)
    video_translation = manager.translate(video_translation, source_lang=source_lang, target_lang=target_lang)
    torch.cuda.empty_cache()

    manager = TextToSpeechManager(public_id, api_client, file_repository, tts_name=tts_model, tts_sample_rate=24000, device="cuda:0", logger=logger, eleven_api_token=eleven_api_token)
    video_translation = manager.synthesize(video_translation, source_lang=source_lang, target_lang=target_lang, voice_conv=True, merge_pipeline=dubbing_algo, enhance=True)
    torch.cuda.empty_cache()
    print("6")
    return f"{base_dir}/{public_id}/resulted_video.mp4"




def gradio_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Video Translation Pipeline")
        gr.Markdown("Upload a video and specify source/target languages for translation. The output will be a translated video.")
        
        with gr.Row():
            source_lang = gr.Dropdown(choices=LANGUAGES_SOURCE, label="Source Language", value="Russian")
            target_lang = gr.Dropdown(choices=LANGUAGES_TARGET, label="Target Language", value="English")
            tts_model = gr.Dropdown(choices=TTS_MODELS, label="tts model", value=TTS_MODELS[0])
            dubbing_algo = gr.Dropdown(choices=DUBBING_ALGO, label="dubbing algorithm", value=DUBBING_ALGO[0])
            eleven_api_token =  gr.Textbox(label="Token for elevenlabs", value="")
            num_speakers = gr.Textbox(label="Number of Speakers", value="1")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video", value="demo.mp4")
                translate_button = gr.Button("Translate Video")
            output_video = gr.Video(label="Translated Video", value="resulted_video.mp4")

        def clear_output():
            return None
        Log(log_file_path, dark=True, xterm_font_size=12)

        video_input.change(fn=clear_output, inputs=[], outputs=output_video)

        translate_button.click(
            fn=video_translation_pipeline,
            inputs=[video_input, source_lang, target_lang, tts_model, dubbing_algo, eleven_api_token, num_speakers],
            outputs=output_video
        )
        
    demo.launch(server_name="0.0.0.0", server_port=4444)

if __name__ == "__main__":
    gradio_demo()
