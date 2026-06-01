"""
Tests for F5 TTS (Flow-based Text-to-Speech) components.
Following the existing testing pattern with fixtures and parametrization.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

# F5-TTS depends on the optional `f5_tts` package; skip the whole module if absent.
pytest.importorskip("f5_tts", reason="F5-TTS backend optional dependency 'f5_tts' not installed")

from langswap.ml.text_to_speech_service.tts_f5_client import FlowClient


@patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
class TestFlowClientInit:
    """Test FlowClient initialization with mocked dependencies."""
    
    def test_init_default_params(self, mock_load_model, mock_load_vocoder, mock_ruaccent):
        """Test initialization with default parameters."""
        # Setup mocks
        mock_vocoder = Mock()
        mock_load_vocoder.return_value = mock_vocoder
        
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_accent_instance = Mock()
        mock_ruaccent.return_value = mock_accent_instance
        
        # Initialize client
        client = FlowClient()
        
        # Verify initialization
        assert client.vocab_file == './models_weights/ESpeech-TTS/vocab.txt'
        assert client.sample_rate == 24000
        assert client.model_path == './models_weights/ESpeech-TTS/model_40000.pt'
        assert client.vocos == mock_vocoder
        assert client.tts == mock_model
        assert client.accentizer == mock_accent_instance
        
        # Verify vocoder loading
        mock_load_vocoder.assert_called_once_with(
            is_local=True, 
            local_path="./models_weights/vocos-mel-24khz"
        )
        
        # Verify accentizer loading
        mock_accent_instance.load.assert_called_once_with(
            omograph_model_size='turbo3.1',
            use_dictionary=True,
            tiny_mode=False,
            workdir="./models_weights/ruaccent"
        )
    
    def test_init_custom_params(self, mock_load_model, mock_load_vocoder, mock_ruaccent):
        """Test initialization with custom parameters."""
        custom_vocab = "/custom/vocab.txt"
        custom_vocos_path = "/custom/vocos"
        custom_model_path = "/custom/model.pt"
        
        mock_vocoder = Mock()
        mock_load_vocoder.return_value = mock_vocoder
        
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_accent_instance = Mock()
        mock_ruaccent.return_value = mock_accent_instance
        
        client = FlowClient(
            vocab_file=custom_vocab,
            vocos_local_path=custom_vocos_path,
            model_path=custom_model_path
        )
        
        assert client.vocab_file == custom_vocab
        assert client.model_path == custom_model_path
        
        mock_load_vocoder.assert_called_once_with(
            is_local=True,
            local_path=custom_vocos_path
        )


@patch('src.ml.text_to_speech_service.tts_f5_client.DiT')
@patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
class TestLoadTTSFlow:
    """Test TTS model loading functionality."""
    
    def test_load_tts_flow_model_config(self, mock_load_model, mock_load_vocoder, mock_ruaccent, mock_dit):
        """Test that model is loaded with correct configuration."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        client = FlowClient()
        
        # Verify model loading parameters
        mock_load_model.assert_called_once()
        call_args = mock_load_model.call_args
        
        assert call_args[1]['model_cls'] == mock_dit
        assert call_args[1]['model_cfg'] == {
            'dim': 1024,
            'depth': 22,
            'heads': 16,
            'ff_mult': 2,
            'text_dim': 512,
            'conv_layers': 4
        }
        assert call_args[1]['ckpt_path'] == client.model_path
        assert call_args[1]['mel_spec_type'] == "vocos"
        assert call_args[1]['vocab_file'] == client.vocab_file
        assert call_args[1]['device'] == "cuda"


@patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
@patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
@patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
class TestGenerateAudio:
    """Test audio generation functionality."""
    
    def test_generate_audio_basic(self, mock_load_model, mock_load_vocoder, mock_ruaccent, 
                                  mock_infer_process, mock_sf_write, tmp_path):
        """Test basic audio generation without special markers."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_vocoder = Mock()
        mock_load_vocoder.return_value = mock_vocoder
        
        mock_accent_instance = Mock()
        mock_accent_instance.process_all.return_value = "processed text"
        mock_ruaccent.return_value = mock_accent_instance
        
        # Mock audio generation
        sample_rate = 24000
        audio_data = np.random.randn(sample_rate * 2).astype(np.float32)
        mock_infer_process.return_value = (audio_data, sample_rate, None)
        
        # Create client and generate audio
        client = FlowClient()
        
        text = "Hello world"
        source_audio = str(tmp_path / "source.wav")
        source_text = "Reference text"
        save_path = str(tmp_path / "output.wav")
        
        client.generate_audio(
            text=text,
            source_audio_file=source_audio,
            source_text=source_text,
            save_path=save_path,
            language="english"
        )
        
        # Verify inference was called
        mock_infer_process.assert_called_once_with(
            ref_audio=source_audio,
            ref_text=source_text,
            gen_text=text,
            model_obj=mock_model,
            vocoder=mock_vocoder,
            mel_spec_type="vocos",
            fix_duration=None
        )
        
        # Verify audio was saved
        mock_sf_write.assert_called_once()
        call_args = mock_sf_write.call_args
        assert call_args[0][1].shape == audio_data.shape
        assert call_args[0][2] == sample_rate
    
    def test_generate_audio_russian_accentization(self, mock_load_model, mock_load_vocoder, 
                                                   mock_ruaccent, mock_infer_process, 
                                                   mock_sf_write, tmp_path):
        """Test Russian text processing with accentizer."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_accent_instance = Mock()
        mock_accent_instance.process_all.return_value = "акцентированный текст"
        mock_ruaccent.return_value = mock_accent_instance
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        client = FlowClient()
        
        text = "привет мир"
        client.generate_audio(
            text=text,
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="russian"
        )
        
        # Verify accentizer was called for Russian
        mock_accent_instance.process_all.assert_called_once_with(text)
        
        # Verify inference used processed text
        call_args = mock_infer_process.call_args
        assert call_args[1]['gen_text'] == "акцентированный текст"
    
    def test_generate_audio_with_markers(self, mock_load_model, mock_load_vocoder, 
                                         mock_ruaccent, mock_infer_process, 
                                         mock_sf_write, tmp_path):
        """Test audio generation with special markers like [laugh], [cough], etc."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Mock multiple audio segments
        audio_segment1 = np.random.randn(12000).astype(np.float32)
        audio_segment2 = np.random.randn(8000).astype(np.float32)
        audio_segment3 = np.random.randn(10000).astype(np.float32)
        
        mock_infer_process.side_effect = [
            (audio_segment1, 24000, None),
            (audio_segment2, 24000, None),
            (audio_segment3, 24000, None)
        ]
        
        client = FlowClient()
        
        text = "Hello [laugh] how are you [cough] today?"
        client.generate_audio(
            text=text,
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="english"
        )
        
        # Verify multiple inference calls for each segment
        assert mock_infer_process.call_count == 3
        
        # Check generated texts for each call
        calls = mock_infer_process.call_args_list
        assert calls[0][1]['gen_text'] == "Hello"
        assert calls[1][1]['gen_text'] == "how are you"
        assert calls[2][1]['gen_text'] == "today?"
        
        # Verify final audio is concatenated
        mock_sf_write.assert_called_once()
        saved_audio = mock_sf_write.call_args[0][1]
        expected_length = len(audio_segment1) + len(audio_segment2) + len(audio_segment3)
        assert len(saved_audio) == expected_length
    
    def test_generate_audio_with_duration(self, mock_load_model, mock_load_vocoder, 
                                          mock_ruaccent, mock_infer_process, 
                                          mock_sf_write, tmp_path):
        """Test audio generation with fixed duration parameter."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        client = FlowClient()
        
        duration = 2.5
        client.generate_audio(
            text="Test text",
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="english",
            duration=duration
        )
        
        # Verify duration was passed to inference
        call_args = mock_infer_process.call_args
        assert call_args[1]['fix_duration'] == duration
    
    def test_generate_audio_empty_text(self, mock_load_model, mock_load_vocoder, 
                                       mock_ruaccent, mock_infer_process, 
                                       mock_sf_write, tmp_path):
        """Test handling of empty text input."""
        client = FlowClient()
        
        # Empty text should not generate audio
        client.generate_audio(
            text="",
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="english"
        )
        
        # No inference should happen
        mock_infer_process.assert_not_called()
        # But file should still be created (empty)
        mock_sf_write.assert_called_once()


@patch('src.ml.text_to_speech_service.tts_f5_client.os.path.exists')
@patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
@patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
@patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
@patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
class TestTTSPipeline:
    """Test TTS pipeline functionality."""
    
    def test_tts_pipeline_basic(self, mock_load_model, mock_load_vocoder, mock_ruaccent,
                                mock_infer_process, mock_sf_write, mock_exists, tmp_path):
        """Test basic TTS pipeline with multiple segments."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_exists.return_value = False  # Files don't exist yet
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        # Create mock video translation
        mock_video_translation = Mock()
        
        # Create segments
        segments = []
        for i in range(3):
            segment = Mock()
            segment.start = float(i * 2)
            segment.end = float(i * 2 + 1.5)
            segment.translation = f"Translation {i}"
            segment.source_file = f"source_{i}.wav"
            segment.text = f"Original {i}"
            segments.append(segment)
        
        mock_video_translation.translated_texts = segments
        
        client = FlowClient()
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path), language="en")
        
        # Verify results
        assert result == mock_video_translation
        assert mock_infer_process.call_count == 3
        
        # Verify each segment has generated file assigned
        for i, segment in enumerate(segments):
            expected_path = os.path.join(str(tmp_path), f"{segment.start}_{segment.end}.wav")
            assert segment.generated_file == expected_path
    
    def test_tts_pipeline_existing_files(self, mock_load_model, mock_load_vocoder, 
                                         mock_ruaccent, mock_infer_process, 
                                         mock_sf_write, mock_exists, tmp_path):
        """Test pipeline skips existing files."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # First file exists, second doesn't
        mock_exists.side_effect = [True, False]
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        # Create mock segments
        mock_video_translation = Mock()
        segments = []
        for i in range(2):
            segment = Mock()
            segment.start = float(i)
            segment.end = float(i + 1)
            segment.translation = f"Translation {i}"
            segment.source_file = f"source_{i}.wav"
            segment.text = f"Original {i}"
            segments.append(segment)
        
        mock_video_translation.translated_texts = segments
        
        client = FlowClient()
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Only one file should be generated (second one)
        assert mock_infer_process.call_count == 1
        
        # Both segments should have file paths assigned
        for segment in segments:
            assert hasattr(segment, 'generated_file')
            assert segment.generated_file is not None
    
    def test_tts_pipeline_russian_language(self, mock_load_model, mock_load_vocoder, 
                                           mock_ruaccent, mock_infer_process, 
                                           mock_sf_write, mock_exists, tmp_path):
        """Test pipeline with Russian language."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_accent_instance = Mock()
        mock_accent_instance.process_all.return_value = "обработанный текст"
        mock_ruaccent.return_value = mock_accent_instance
        
        mock_exists.return_value = False
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        # Create mock segment
        mock_video_translation = Mock()
        segment = Mock()
        segment.start = 0.0
        segment.end = 1.0
        segment.translation = "Привет мир"
        segment.source_file = "source.wav"
        segment.text = "Hello world"
        
        mock_video_translation.translated_texts = [segment]
        
        client = FlowClient()
        client.tts_pipeline(mock_video_translation, str(tmp_path), language="russian")
        
        # Verify accentizer was called
        mock_accent_instance.process_all.assert_called_once_with("Привет мир")
    
    @pytest.mark.parametrize("language", ["en", "es", "fr", "de", "russian"])
    def test_tts_pipeline_language_handling(self, mock_load_model, mock_load_vocoder, 
                                            mock_ruaccent, mock_infer_process, 
                                            mock_sf_write, mock_exists, language, tmp_path):
        """Test pipeline handles different languages correctly."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_accent_instance = Mock()
        mock_accent_instance.process_all.return_value = "processed"
        mock_ruaccent.return_value = mock_accent_instance
        
        mock_exists.return_value = False
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        mock_video_translation = Mock()
        segment = Mock()
        segment.start = 0.0
        segment.end = 1.0
        segment.translation = "Test"
        segment.source_file = "source.wav"
        segment.text = "Test"
        
        mock_video_translation.translated_texts = [segment]
        
        client = FlowClient()
        client.tts_pipeline(mock_video_translation, str(tmp_path), language=language)
        
        # Accentizer should only be called for Russian
        if language == "russian":
            mock_accent_instance.process_all.assert_called_once()
        else:
            mock_accent_instance.process_all.assert_not_called()


class TestFlowClientEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    def test_model_loading_failure(self, mock_load_vocoder, mock_load_model):
        """Test handling of model loading failure."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception) as exc_info:
            FlowClient()
        
        assert "Model loading failed" in str(exc_info.value)
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_inference_failure(self, mock_load_model, mock_load_vocoder, mock_ruaccent,
                               mock_infer_process, mock_sf_write, tmp_path):
        """Test handling of inference failure."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_infer_process.side_effect = Exception("Inference failed")
        
        client = FlowClient()
        
        with pytest.raises(Exception) as exc_info:
            client.generate_audio(
                text="Test",
                source_audio_file="source.wav",
                source_text="ref",
                save_path=str(tmp_path / "output.wav"),
                language="english"
            )
        
        assert "Inference failed" in str(exc_info.value)
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_special_characters_in_text(self, mock_load_model, mock_load_vocoder, 
                                        mock_ruaccent, mock_infer_process, 
                                        mock_sf_write, tmp_path):
        """Test handling of special characters in text."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        client = FlowClient()
        
        # Text with various special characters
        text = "Hello! How are you? I'm fine... Really @ #special $text%"
        client.generate_audio(
            text=text,
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="english"
        )
        
        # Should process without error
        mock_infer_process.assert_called_once()
        mock_sf_write.assert_called_once()
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.os.path.exists')
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_tts_pipeline_empty_segments(self, mock_load_model, mock_load_vocoder, 
                                         mock_ruaccent, mock_infer_process, 
                                         mock_sf_write, mock_exists, tmp_path):
        """Test pipeline with empty segment list."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_video_translation = Mock()
        mock_video_translation.translated_texts = []
        
        client = FlowClient()
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Should handle empty list gracefully
        assert result == mock_video_translation
        mock_infer_process.assert_not_called()


class TestFlowClientIntegration:
    """Integration tests for complete workflows."""
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.os.path.exists')
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_complete_workflow_with_markers(self, mock_load_model, mock_load_vocoder, 
                                            mock_ruaccent, mock_infer_process, 
                                            mock_sf_write, mock_exists, tmp_path):
        """Test complete workflow with text containing markers."""
        # Setup comprehensive mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_vocoder = Mock()
        mock_load_vocoder.return_value = mock_vocoder
        
        mock_accent_instance = Mock()
        mock_accent_instance.process_all.side_effect = lambda x: f"accented_{x}"
        mock_ruaccent.return_value = mock_accent_instance
        
        mock_exists.return_value = False
        
        # Different audio segments for variety
        audio_segments = [
            np.random.randn(20000).astype(np.float32),
            np.random.randn(15000).astype(np.float32),
            np.random.randn(25000).astype(np.float32)
        ]
        
        mock_infer_process.side_effect = [
            (seg, 24000, None) for seg in audio_segments
        ]
        
        # Create complex video translation
        mock_video_translation = Mock()
        segments = []
        
        segment1 = Mock()
        segment1.start = 0.0
        segment1.end = 2.0
        segment1.translation = "Hello [laugh] world"
        segment1.source_file = "source1.wav"
        segment1.text = "Original 1"
        segments.append(segment1)
        
        segment2 = Mock()
        segment2.start = 2.0
        segment2.end = 4.0
        segment2.translation = "Привет мир"  # Russian text
        segment2.source_file = "source2.wav"
        segment2.text = "Original 2"
        segments.append(segment2)
        
        mock_video_translation.translated_texts = segments
        
        client = FlowClient()
        
        # Test English pipeline
        result = client.tts_pipeline(mock_video_translation, str(tmp_path), language="en")
        
        # Verify segment 1 generated multiple audio parts due to [laugh] marker
        assert mock_infer_process.call_count >= 2
        
        # Test Russian pipeline
        mock_infer_process.reset_mock()
        mock_infer_process.side_effect = [
            (seg, 24000, None) for seg in audio_segments
        ]
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path), language="russian")
        
        # Verify Russian text was processed
        assert mock_accent_instance.process_all.called
        
        # Verify all segments have generated files
        for segment in segments:
            assert hasattr(segment, 'generated_file')
            assert segment.generated_file is not None


# Performance and stress tests
class TestFlowClientPerformance:
    """Test performance-related aspects."""
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.os.path.exists')
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_large_batch_processing(self, mock_load_model, mock_load_vocoder, 
                                    mock_ruaccent, mock_infer_process, 
                                    mock_sf_write, mock_exists, tmp_path):
        """Test processing large number of segments."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_exists.return_value = False
        
        audio_data = np.random.randn(24000).astype(np.float32)
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        # Create 100 segments
        mock_video_translation = Mock()
        segments = []
        for i in range(100):
            segment = Mock()
            segment.start = float(i * 2)
            segment.end = float(i * 2 + 1.5)
            segment.translation = f"Translation {i}"
            segment.source_file = f"source_{i}.wav"
            segment.text = f"Original {i}"
            segments.append(segment)
        
        mock_video_translation.translated_texts = segments
        
        client = FlowClient()
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Verify all segments processed
        assert mock_infer_process.call_count == 100
        assert all(hasattr(seg, 'generated_file') for seg in segments)
    
    @patch('src.ml.text_to_speech_service.tts_f5_client.sf.write')
    @patch('src.ml.text_to_speech_service.tts_f5_client.infer_process')
    @patch('src.ml.text_to_speech_service.tts_f5_client.RUAccent')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_vocoder')
    @patch('src.ml.text_to_speech_service.tts_f5_client.load_model')
    def test_very_long_text(self, mock_load_model, mock_load_vocoder, 
                            mock_ruaccent, mock_infer_process, 
                            mock_sf_write, tmp_path):
        """Test handling of very long text input."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Generate very long audio
        audio_data = np.random.randn(24000 * 60).astype(np.float32)  # 60 seconds
        mock_infer_process.return_value = (audio_data, 24000, None)
        
        client = FlowClient()
        
        # Very long text
        long_text = " ".join(["This is a test sentence."] * 100)
        
        client.generate_audio(
            text=long_text,
            source_audio_file="source.wav",
            source_text="ref",
            save_path=str(tmp_path / "output.wav"),
            language="english"
        )
        
        # Should handle without error
        mock_infer_process.assert_called_once()
        mock_sf_write.assert_called_once()
