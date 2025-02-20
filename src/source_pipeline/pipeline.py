import os
import json
import torchaudio
import spacy
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import Audio
import ffmpeg
import torch
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from google.cloud import speech
import whisperx
import deepl
from pydub import AudioSegment
from torchaudio import transforms
from pyrubberband.pyrb import time_stretch
import numpy as np
import requests
import demucs.api

from omegaconf import DictConfig
import hydra
import logging

log = logging.getLogger(__name__)


from utils_audio import resample_save, prepare_split

class PipelineVideoTranslate:
    def __init__(self, cfg):
        self.cfg = cfg

        self.SAMPLING_RATE = 16000
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temp_folder = self.cfg.temp_dir
        self.GCS_BUCKET = "peacedata_user_audios"
        self.GOOGLE_APPLICATION_CREDENTIALS = "./peacedata-tts/src/gcp/gcp_key.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.GOOGLE_APPLICATION_CREDENTIALS

        self.client_elevenlabs = ElevenLabs(api_key=self.cfg.elevenlabs_api_key)
        self.model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        (self.vad_get_speech_timestamps, self.vad_save_audio, self.vad_read_audio, self.VADIterator, self.vad_collect_chunks) = utils

        self.translator = deepl.Translator("1a9bfdf3-17d8-4ffa-bc00-54e4249506cd:fx")
        self.nlp = spacy.load("xx_sent_ud_sm") 
    
    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def extract_video_from_audio(self, video_fname):
        video_fname_wo_ext = video_fname.rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
        out_path = os.path.join(self.temp_folder, f'{video_fname_wo_ext}.wav')
        (
            ffmpeg
            .input(video_fname)
            .output(out_path)
            .run(overwrite_output=True)
        )
        return out_path, video_fname_wo_ext
    
    def vad_filter(self, audio_path):
        wav = self.vad_read_audio(audio_path, sampling_rate=self.SAMPLING_RATE)
        speech_timestamps = self.vad_get_speech_timestamps(wav, self.model_vad, sampling_rate=self.SAMPLING_RATE, return_seconds=False)
        self.vad_save_audio(audio_path, self.vad_collect_chunks(speech_timestamps, wav), sampling_rate=self.SAMPLING_RATE) 
        return audio_path

    def transcribe_audio(self, audio_full_path, whisper_name="medium"):
        asr = whisperx.load_model(whisper_name, device=self.DEVICE, compute_type='int8')
        transcribed_text = asr.transcribe(audio_full_path)
        model_a, metadata = whisperx.load_align_model(language_code=transcribed_text["language"], device=self.DEVICE)
        result_aligned = whisperx.align(transcribed_text["segments"], model_a, metadata, audio_full_path, self.DEVICE)

        transcribed_text_path = os.path.join(self.cfg.temp_dir, "transcribed_text.json")
        
        with open(transcribed_text_path, 'w', encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=4)

        log.info(f"Transcribed text is saved in the file: {transcribed_text_path}")
        return result_aligned

    def load_spacy_model(self, language='xx'):
        spacy_languages = {
            'en': "en_core_web_sm",
            'ru': "ru_core_news_sm",
            'fr': "fr_core_news_sm",
            'zh': "zh_core_web_sm",
            "de": "de_core_news_sm",
            "nl": "nl_core_news_sm",
            "pl": "pl_core_news_sm",
            "xx": "xx_sent_ud_sm"  # multilingual model
        }
        selected_model = spacy_languages[language]
        try:
            nlp = spacy.load(selected_model) 
        except OSError:
            spacy.cli.download(selected_model)
            nlp = spacy.load(selected_model) 
        return nlp

    def remap_sentences(self, transcribed_text):
        nlp = self.load_spacy_model()
        plain_text = ' '.join([x['word'] for x in transcribed_text['word_segments']])
        doc = nlp(plain_text)
        sent_bounds = [x[0].idx for x in doc.sents]
        df_words = pd.DataFrame(transcribed_text['word_segments'])
        df_words['text'] = df_words.word
        df_words['len'] = df_words.text.apply(len)
        df_words['end_pos'] = (df_words['len'] + 1).cumsum()
        df_words['start_pos'] = df_words['end_pos'].shift(1, fill_value=0)
        for i, x in enumerate(sent_bounds):
            df_words.loc[df_words['end_pos'] > x, 'sent'] = i
        df_words.sent = df_words.sent.astype(int)
        sentences = []
        for i in df_words.sent.unique():
            entry = {}
            slc = df_words.loc[df_words.sent == i]
            entry['text'] = ' '.join(slc.text.to_list())
            entry['start'] = slc.start.min()
            entry['end'] = slc.end.max()
            sentences.append(entry)
        df = pd.DataFrame(sentences)
        return df
    
    def create_voice(self, voice_path, voice_name="Dance voice", description="Young lady with soft voice."):
        voice = self.client_elevenlabs.clone(voice_name, description=description,
                                    files=[voice_path])
        return voice

    def generate_translation_audio(self, df, voice):
        for i, line in tqdm(df.iterrows(), total=df.shape[0]):
            audio = self.client_elevenlabs.generate(text=line.translation, voice=voice)
            generated_audio_path = os.path.join(self.cfg.temp_dir, f"{i}.wav")
            save(audio, generated_audio_path)
            df.loc[i, 'syn_audio_path'] = generated_audio_path

        df['gen_dur'] = df['syn_audio_path'].apply(lambda x: torchaudio.load(x)[0].shape[1] / self.cfg.tts_sample_rate)
        df['pause'] = df['start'].shift(-1) - df['end']
        df['dur_gen_pause'] = df['gen_dur'] + df['pause']
        df['place_gen'] = df['end'] - df['start'] + df['pause']
        df['gen_end'] = df['start'] + df['gen_dur']
        df['can_start'] = [0] + df['gen_end'].to_list()[:-1]
        df['need_time'] = df['gen_dur'] - df['place_gen']
        df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        df['need_speedup'] = df['gen_dur'] > df['place_gen']
        df['duration_orig'] = df['end'] - df['start']
        return df

    def merge_audio_timestamps(self, df, video_length, sr=44100):
        blank_audio_tensor = torch.zeros((1, int(video_length * sr)))

        for i, line in tqdm(df.iterrows(), total=df.shape[0]):
            wav, sr = torchaudio.load(line.syn_audio_path)
            start_pos = line.new_start * self.cfg.tts_sample_rate
            end_pos = start_pos + wav.shape[-1]
            start_pos = np.ceil(start_pos)
            end_pos = np.ceil(end_pos)
            blank_audio_tensor[0, int(start_pos): int(end_pos)] = wav[0]

        generated_audio_path = os.path.join(self.cfg.temp_dir, "merged_audio.wav")
        torchaudio.save(generated_audio_path, blank_audio_tensor, sample_rate=sr)
        log.info(f" Merged audio shape is {blank_audio_tensor.shape}.")
        return blank_audio_tensor, generated_audio_path
    
    def merge_background(self, audio_file_path, clean_audio_path, temp_folder):
        separator = demucs.api.Separator()
        separated = separator.separate_audio_file(audio_file_path)

        for file, source in separated[1].items():
            if file == "other":
                save_path = os.path.join(temp_folder, f"{file}.wav")
                demucs.api.save_audio(source, save_path, samplerate=separator.samplerate)
        
        background_sound, sr_back = torchaudio.load(clean_audio_path)
        speech_audio, sr_speech = torchaudio.load(save_path)

        assert sr_back == sr_speech, "Background sr is not equal to speech sr."

        common_sound = background_sound + speech_audio
        save_path = os.path.join(temp_folder, f"merged_background.wav")
        torchaudio.save(save_path, common_sound, sample_rate=self.cfg.tts_sample_rate)
        return save_path
        

    def replace_audio_in_video(self, input_video_name, generated_full_audio, translated_video_folder):
        video_id = input_video_name.rsplit('/', maxsplit=1)[-1].split('.')[0]
        audio = ffmpeg.input(generated_full_audio).audio
        video = ffmpeg.input(input_video_name).video
        out = ffmpeg.output(audio, video, os.path.join(translated_video_folder, f'{video_id}.mp4'), acodec='aac').run()
        return f'translated_videos/{video_id}.mp4'
    
    def run_pipeline(self):
        # pipeline steps:
        # 0. create a temp forlder for all files 
        # 1. extract video from audio 
        # 2. sample the first minute 
        # 3. extract the vad filtering 
        # 4. create voice with eleven labs 
        # 5. transcribe audio 
        # 6. remap the transcribed sentennces
        # 7. translate the texts 
        # 8. generate audio 
        # 9. get the background sound from the video 
        # 10. merge the background sound with generated translated audio 
        # 11. replace audio in video 

        audio_full_path, _ = self.extract_video_from_audio(self.cfg.video_path)
        log.info(f"PIPELINE STAGE 1: {audio_full_path} was extracted from the video.")

        # Prepare and process the audio            
        first_minute_path = prepare_split(audio_full_path, seconds_start=self.cfg.seconds_start, seconds_end=self.cfg.seconds_end)
        first_minute_path = resample_save(first_minute_path)
        first_minute_path = self.vad_filter(first_minute_path)
        log.info(f"PIPELINE STAGE 2-3: {first_minute_path} is the first minute extracted from the video.")

        voice = self.create_voice(first_minute_path)
        log.info(f"PIPELINE STAGE 4: Voice was created.")

        # # Transcribe the audio
        audio_full_path_resampled = resample_save(audio_full_path)
        transcribed_text = self.transcribe_audio(audio_full_path_resampled)
        log.info(f"PIPELINE STAGE 5: Voice was created.")

        # Remap sentences
        df = self.remap_sentences(transcribed_text)
        log.info(f"PIPELINE STAGE 6: Remapped the sentences.")

        # df = pd.read_csv("/Users/Milana/Documents/code-projects/tts_multilingual/df_dance_lesson.csv")

        translations = self.translator.translate_text([x for x in df['text']], target_lang='EN-US')
        df['translation'] = [x.text for x in translations]
        log.info(f"PIPELINE STAGE 7: Translated the text.")

        df = self.generate_translation_audio(df, voice)
        log.info(f"PIPELINE STAGE 8: Generate the voice in target lang.")
        df_path = os.path.join(self.cfg.temp_dir, "df_base.csv")
        df.to_csv(df_path, index=False)

        # Merge audio with video
        video_length = torchaudio.info(audio_full_path).num_frames / torchaudio.info(audio_full_path).sample_rate
        merged_audio, merged_audio_path = self.merge_audio_timestamps(df, video_length)
        log.info(f"PIPELINE STAGE 9: Merged audio and video in the target lang in {merged_audio_path}.")

        merged_background_audio = self.merge_background(audio_full_path, merged_audio_path, self.cfg.temp_dir)
        log.info(f"PIPELINE STAGE 10: Merge the backgound.")

        # Replace audio in video
        self.replace_audio_in_video(self.cfg.video_path, merged_background_audio, self.cfg.translated_video_folder)
        log.info(f"PIPELINE STAGE 11: Added the new video in the target lang.")


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pipeline = PipelineVideoTranslate.from_config(cfg)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

