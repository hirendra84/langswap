import pandas as pd
import torchaudio
import torch
import numpy as np
from langswap.file_repository import FileRepository
from pyrubberband.pyrb import time_stretch
import pandas as pd
import numpy as np
import torchaudio.transforms as T
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


class VideoDubbingManager:
    _file_repository: FileRepository

    def __init__(self, file_repository: FileRepository, logger):
        self._file_repository = file_repository

        self.logger = logger

        self.model_vad = load_silero_vad()
    
    def get_pause(self, wav_path, sr, seconds=True):
        """
        Calculates the intra pauses in the audio
        """

        # hardcoded - change before that
        if "enhanced_audio" in wav_path:
            wav_path = wav_path.replace("enhanced_audio", "splitted_audio")
            wav_path = wav_path.replace("_enhanced", "")

        wav = read_audio(wav_path)
        speech_timesteps = get_speech_timestamps(wav, self.model_vad, return_seconds=seconds)
        # print(f"{wav_path}, {len(speech_timesteps)} speech timesteps are {speech_timesteps}")
        pause_long = 0
        # where does it happen that we do not have speech timesteps? 
        if speech_timesteps:
            speech_timesteps[0]['pause'] = 0
            for t_idx in range(1, len(speech_timesteps)):
                speech_timesteps[t_idx]['pause'] = speech_timesteps[t_idx]['start'] - speech_timesteps[t_idx - 1]['end']

            pause_long = sum([i['pause'] for i in speech_timesteps])
        
        return pause_long, speech_timesteps
    
    def merge_pauses(self, wav_path, sr, timesteps):
        """
        Merges the audio back with pauses according to the timesteps
        """
        wav = read_audio(wav_path, sampling_rate=sr)

        # audio_samples = [wav[timesteps[0]['start']: timesteps[0]['end'] + 1].unsqueeze(0)]
        start = int(timesteps[0]['start'] * sr)
        end = int((timesteps[0]['end']) * sr)
        audio_samples = [wav[start: end].unsqueeze(0)]

        for audio_info in timesteps[1:]:
            pause = torch.zeros((1, int(audio_info['pause'])))
            audio_samples.append(pause)

            audio = wav[int(audio_info['start'] * sr): int((audio_info['end']) * sr)].unsqueeze(0)
            audio_samples.append(audio)
        
        audio_final = torch.cat(audio_samples, dim=1)
        self.logger.file_logger.debug(f"Changed audio length is from {wav.shape[0] / sr} to {audio_final.shape[1] / sr} for file {wav_path}")

        torchaudio.save(wav_path, audio_final, sr)
        return audio_final
    
    def change_pauses(self, wav_gen_path: str, wav_source_path: str,
                    sr_gen: int, sr_source: int):
        """
        1. Calculates the pause duration of source and geneerated, 
        2. Compares the pause of the two
        3. Changes the duration of the pauses
        4. Merges generated wav into a new wav
        """
        # SOURCE FILE HERE IS INCORRECT
        pause_dur_source, timesteps_source = self.get_pause(wav_source_path, sr_source, seconds=True)

        pause_dur_gen, timesteps_gen = self.get_pause(wav_gen_path, sr_gen, seconds=True)
        if pause_dur_gen:
            # TODO: the main trick is here to calculate the pause reduction and when to actually apply it
            pause_reduction = (pause_dur_gen - pause_dur_source) / pause_dur_gen
            if pause_reduction == 1.0:
                pause_reduction = 0.8
            # _, timesteps_gen = self.get_pause(wav_gen_path, sr=sr_gen, seconds=False)
            for sp_idx, sp_data in enumerate(timesteps_gen):
                # if pause_reduction < 1: # TODO: change not to make a pass
                sp_data['pause'] = sp_data['pause']  - (pause_reduction * sp_data['pause'])

            audio = self.merge_pauses(wav_gen_path, sr_gen, timesteps_gen)
        else:
            audio, sr_gen = torchaudio.load(wav_gen_path)
        return audio
    
    def merge_timestamps_speedup(self, video_translation, vocals_audio):
        """
        The algo to speed up each of the samples:
        1. 
        1. Change the intra pauses. 
        2. If the audio is still longer -> speed it up
        3. If the audio is shorter -> add additional pause to start at the correct time slot.
        4. Merge all of the samples. If the audio is longer -> speed up the whole audio, should not be the case here.
        """
        df_dict = {"start": [], "end": []}

        # fill in the df
        for segment in video_translation.translated_texts:
            df_dict["start"].append(segment.start)
            df_dict["end"].append(segment.end)

        df = pd.DataFrame.from_dict(df_dict)
        df["pause"] = df["start"].shift(-1) - df["end"] # pause between two samples
        df["source_length"] = df["end"] - df["start"] # full audio length
        df["raw_audio_length"] = [0.0] * df.shape[0] 
        df["generated_audio_length"] = [0.0] * df.shape[0]
        df["generated_audio_length_with_pause"] = [0.0] * df.shape[0]
        df["generated_audio_pause"] = [0.0] * df.shape[0]
        df["generated_end"] = [0.0] * df.shape[0]
        
        # calculate the last pause length
        source_audio, source_sr = torchaudio.load(vocals_audio)
        target_audio_length = source_audio.shape[1] / source_sr

        if df.shape[0] > 1:
            df.loc[df.shape[0] - 1, "pause"] = target_audio_length - df.loc[df.shape[0] - 1, "end"]
        elif df.shape[0] == 1:
            df.loc[0, "pause"] = 0

        _, sr_generated = torchaudio.load(video_translation.translated_texts[0].generated_file)
        first_pause = torch.zeros((1, int(video_translation.translated_texts[0].start * sr_generated)))

        audio_generated = first_pause
        for idx, segment in enumerate(video_translation.translated_texts):
            pause_size = df.loc[idx, "pause"]

            audio, sr_generated = torchaudio.load(segment.generated_file)
            audio_length = audio.shape[1] / sr_generated

            df.loc[idx, "raw_audio_length"] = audio_length
            source_length = df["source_length"].iloc[idx]
            audio = self.change_pauses(segment.generated_file, segment.source_file,
                                    sr_gen=sr_generated, sr_source=source_sr)
                
            audio_length = audio.shape[1] / sr_generated
            
            pause_size_to_take = 0.5 * pause_size
            # print(f"""
            #     available pause size {pause_size},
            #     can extend to a pause with size {pause_size_to_take}""")
            if audio_length > source_length:
                # extend the pause
                fill_in_space = source_length + pause_size_to_take

                pause_size_to_take = min(audio_length - source_length, pause_size_to_take)
                pause_size_to_take = max(pause_size_to_take, 0.0)

                self.logger.file_logger.debug(f"Changing the pause from {pause_size}")

                audio_length -= pause_size_to_take # нужно ускорять в соответствии с измененной длиной 
                pause_size -= pause_size_to_take 

                self.logger.file_logger.debug(f"Changing the pause to {pause_size} with taking additional pause {pause_size_to_take}")

                # if here 
                if audio_length > (source_length + pause_size_to_take):
                    # speed up the audio if the original was still shorter
                    rate = audio_length / (source_length + pause_size_to_take)
                    self.logger.file_logger.debug(f"""Changed audio length with pauses is still longer,
                        generated audio length is {audio_length},
                        source + pause {source_length + pause_size},
                        rate for speed up is {rate}""")
            
                    audio = time_stretch(audio.squeeze().numpy(), sr_generated, rate=rate)
                    audio = torch.tensor(audio).unsqueeze(0)
            
            generated_audio_length = audio.shape[1] / sr_generated
            df.loc[idx, "generated_audio_length"] = generated_audio_length

            # recalculating the pause length -> shorter + original pause
            # if generated_audio_length > source_length:
            #     pause_size -= (generated_audio_length - source_length)
            
            # pause length
            pause_extension = max(source_length - generated_audio_length, 0.0)
            final_pause_length = max(pause_size + pause_extension, 0.0)
            pause = torch.zeros((1, int(final_pause_length * sr_generated)))

            df.loc[idx, "generated_audio_pause"] = pause.shape[1] / sr_generated
            df.loc[idx, "generated_end"] = df.loc[idx, 'start'] + generated_audio_length + pause_extension 

            audio_generated = torch.cat((audio_generated, audio, pause), dim=1)
        
        # final speech speed up rate
        generated_audio_length = audio_generated.shape[1] / sr_generated
        speed_up_rate = generated_audio_length / target_audio_length

        if speed_up_rate > 1:
            audio_generated = time_stretch(audio_generated.squeeze().numpy(), sr=sr_generated, rate=speed_up_rate)
            audio_generated = torch.tensor(audio_generated).unsqueeze(0)

        self.logger.file_logger.debug(f"Rate for speed up is {speed_up_rate}")
        self.logger.log_json(data=df.to_dict('records'), file_name="pauses_for_stretch_whole.json")
        return audio_generated, sr_generated
    
    def merge_timesteps_pause_based(self, video_translation, vocals_audio):
        """
        Algorithm that works on filling the pauses - not the best one.
        """
        df = pd.DataFrame()
        df["start"] = [segment.start for segment in video_translation.translated_texts] 
        df["end"] = [segment.end for segment in video_translation.translated_texts] 
        df["pause"] = df["start"].shift(-1) - df["end"]
        df['gen_dur'] = [torchaudio.load(segment.generated_file)[0].shape[1] / 24000 for segment in video_translation.translated_texts]
        df['pause'] = df['start'].shift(-1) - df['end']
        df['dur_gen_pause'] = df['gen_dur'] + df['pause']
        df['place_gen'] = df['end'] - df['start'] + df['pause']
        df['gen_end'] = df['start'] + df['gen_dur']
        df['can_start'] = [0] + df['gen_end'].to_list()[:-1]
        df['need_time'] = df['gen_dur'] - df['place_gen']
        df['new_start'] = df.apply(lambda x: x.start - x.need_time if x.need_time > 0 else x.start, axis=1)
        df['need_speedup'] = df['gen_dur'] > df['place_gen']
        df['duration_orig'] = df['end'] - df['start']

        prev_audio, sr = torchaudio.load(vocals_audio)
        prev_audio_shape = prev_audio.shape[1]        
        audio_generated = torch.zeros((1, int(prev_audio_shape)))

        for idx, segment in enumerate(video_translation.translated_texts):
            audio, sr = torchaudio.load(segment.generated_file)
            start_pos = df.loc[idx, 'new_start'] * sr

            start_pos = np.ceil(start_pos)
            end_pos = start_pos + audio.shape[-1]
            end_pos = np.ceil(end_pos)

            audio_generated[0, int(start_pos): int(end_pos)] = audio[0]

        return audio_generated, sr

    def merge_timestamps_stretch_whole(self, video_translation, vocals_audio):
        df = pd.DataFrame()
        df["start"] = [segment.start for segment in video_translation.translated_texts] 
        df["end"] = [segment.end for segment in video_translation.translated_texts]
        df["texts"] = [segment.translation for segment in video_translation.translated_texts]

        df["pause"] = df["start"].shift(-1) - df["end"] # pause between two samples 

        df["generated_audio_length"] = [0] * df.shape[0]
        df["generated_audio_pause"] = [0] * df.shape[0]

        df["generated_audio_length"] = df["generated_audio_length"].astype(float)
        df["generated_audio_pause"] = df["generated_audio_pause"].astype(float)
        

        df["generated_start"] = [0.0] * df.shape[0]
        df["generated_end"] = [0.0] * df.shape[0]

        if df.shape[0] > 1:
            # calculate the last pause length
            source_audio, source_sr = torchaudio.load(vocals_audio)
            target_audio_length = source_audio.shape[1] / source_sr
            df.loc[df.shape[0] - 1, "pause"] = target_audio_length - df.loc[df.shape[0] - 1, "end"] # the last pause
        elif df.shape[0] == 1:
            source_audio, source_sr = torchaudio.load(vocals_audio)
            target_audio_length = source_audio.shape[1] / source_sr           
            df.loc[0, "pause"] = 0

        audio_first, sr_generated = torchaudio.load(video_translation.translated_texts[0].generated_file)
        previous_pause = torch.zeros((1, int(video_translation.translated_texts[0].start * sr_generated)))

        audio_generated = previous_pause

        df["source_length"] = df["end"] - df["start"]

        for idx, segment in enumerate(video_translation.translated_texts):
            audio, sr_generated = torchaudio.load(segment.generated_file)
            # audio = self.change_pauses(segment.generated_file, segment.source_file,
            #                         sr_gen=sr_generated, sr_source=source_sr)

            audio_length = audio.shape[1] / sr_generated

            # сократить паузу если необходимо
            time_dif = df["source_length"].iloc[idx] - audio_length

            df.loc[idx, "time_dif"] = time_dif
            
            if time_dif < 0:
                # TODO: check that it is not negative (!)
                df.loc[idx, "pause"] -= time_dif
            else:
                df.loc[idx, "pause"] += time_dif

            pause = torch.zeros((1, int(df.loc[idx, "pause"] * sr_generated)))

            df.loc[idx, "generated_audio_length"] = audio_length
            df.loc[idx, "generated_audio_pause"] = pause.shape[1] / sr_generated

            audio_generated = torch.cat((audio_generated, audio, pause), dim=1)

        generated_audio_length = audio_generated.shape[1] / sr_generated
        speed_up_rate = generated_audio_length/target_audio_length

        self.logger.file_logger.debug(f"Rate for speed up is {speed_up_rate}")
        if speed_up_rate > 1:
            audio_generated = time_stretch(audio_generated.squeeze().numpy(), sr=sr_generated, rate=speed_up_rate)
            audio_generated = torch.tensor(audio_generated).unsqueeze(0)

        data_json = df.to_dict('records')
        self.logger.log_json(data=data_json, file_name="pauses_for_stretch_whole.json")
        return audio_generated, sr_generated