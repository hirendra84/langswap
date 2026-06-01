"""
Tests for TTS (Text-to-Speech) components.
Following the albumentations testing pattern with fixtures and parametrization.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from pathlib import Path

# XTTS depends on the optional `coqui` package; skip the whole module if absent.
pytest.importorskip("coqui", reason="XTTS backend optional dependency 'coqui' not installed")

from langswap.ml.text_to_speech_service.tts_xtts_client import XTTSClient


@patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
class TestXTTSClient:
    """Test XTTSClient with mocked dependencies."""
    
    def test_init_default_params(self, mock_tts_class, mock_file_repository):
        """Test initialization with default parameters."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        assert client.device == "cuda"
        assert client.sample_rate == 24000
        assert client._file_repository == mock_file_repository
        mock_tts_class.assert_called_once()
    
    def test_init_custom_params(self, mock_tts_class, mock_file_repository):
        """Test initialization with custom parameters."""
        custom_model_path = "/custom/model/path"
        custom_device = "cpu"
        
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(
            file_repository=mock_file_repository,
            tts_model_path=custom_model_path,
            device=custom_device
        )
        
        assert client.device == custom_device
        assert custom_model_path in client.tts_model_path
    
    def test_load_models(self, mock_tts_class, mock_file_repository):
        """Test model loading."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        client.load_models()
        
        assert client.model == mock_tts_instance
        mock_tts_instance.to.assert_called_with(client.device)
    
    def test_context_manager(self, mock_tts_class, mock_file_repository):
        """Test XTTSClient as context manager."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        with XTTSClient(file_repository=mock_file_repository) as client:
            assert client.model is not None
        
        # After exit, model should be cleared
        assert client.model is None
    
    @pytest.mark.parametrize("language", ["en", "es", "fr", "de", "it"])
    def test_generate_audio_different_languages(self, mock_tts_class, language, mock_file_repository, sample_text, tmp_path):
        """Test audio generation with different languages."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        source_audio = tmp_path / "source.wav"
        source_audio.touch()
        save_path = tmp_path / "output.wav"
        
        client.generate_audio(
            text=sample_text,
            source_audio_file=str(source_audio),
            source_text="source text",
            save_path=str(save_path),
            language=language
        )
        
        # Verify TTS was called with correct parameters
        mock_tts_instance.tts_to_file.assert_called_once()
        call_args = mock_tts_instance.tts_to_file.call_args
        assert call_args[1]['text'] == sample_text
        assert call_args[1]['language'] == language
        assert call_args[1]['file_path'] == str(save_path)
        assert call_args[1]['speaker_wav'] == str(source_audio)
    
    def test_generate_audio_parameters(self, mock_tts_class, mock_file_repository, sample_text, tmp_path):
        """Test audio generation parameters."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        source_audio = tmp_path / "source.wav"
        source_audio.touch()
        save_path = tmp_path / "output.wav"
        
        client.generate_audio(
            text=sample_text,
            source_audio_file=str(source_audio),
            source_text="",
            save_path=str(save_path),
            language="en"
        )
        
        # Verify specific TTS parameters
        call_args = mock_tts_instance.tts_to_file.call_args
        assert call_args[1]['enable_text_splitting'] is False
        assert call_args[1]['repetition_penalty'] == 2.0


@patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
@patch('src.ml.text_to_speech_service.tts_xtts_client.add_pauses')
@patch('src.ml.text_to_speech_service.tts_xtts_client.merge_speaker_files')
class TestTTSPipeline:
    """Test TTS pipeline functionality."""
    
    def test_tts_pipeline_basic(self, mock_merge, mock_add_pauses, mock_tts_class, mock_file_repository, sample_segments, tmp_path):
        """Test basic TTS pipeline functionality."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create mock video translation
        mock_video_translation = Mock()
        mock_video_translation.translated_texts = sample_segments[:2]  # Use first 2 segments
        
        # Setup segments with required attributes
        for i, segment in enumerate(mock_video_translation.translated_texts):
            segment.start = float(i)
            segment.end = float(i + 1)
            segment.translation = f"Translation {i}"
            segment.source_file = str(tmp_path / f"source_{i}.wav")
            segment.speaker = f"SPEAKER_0{i}"
            # Create source files
            Path(segment.source_file).touch()
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path), language="en")
        
        assert result == mock_video_translation
        # Verify TTS was called for each segment
        assert mock_tts_instance.tts_to_file.call_count == len(mock_video_translation.translated_texts)
    
    def test_tts_pipeline_short_segments(self, mock_merge, mock_add_pauses, mock_tts_class, mock_file_repository, tmp_path):
        """Test TTS pipeline with short segments (< 4 seconds)."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create mock video translation with short segment
        mock_video_translation = Mock()
        short_segment = Mock()
        short_segment.start = 0.0
        short_segment.end = 2.0  # Less than 4 seconds
        short_segment.translation = "Short translation"
        short_segment.source_file = str(tmp_path / "source.wav")
        short_segment.speaker = "SPEAKER_00"
        
        mock_video_translation.translated_texts = [short_segment]
        
        # Create source file
        Path(short_segment.source_file).touch()
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Verify merge_speaker_files was called for short segment
        mock_merge.assert_called_once()
        assert mock_tts_instance.tts_to_file.call_count == 1
    
    def test_tts_pipeline_long_segments(self, mock_merge, mock_add_pauses, mock_tts_class, mock_file_repository, tmp_path):
        """Test TTS pipeline with long segments (>= 4 seconds)."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create mock video translation with long segment
        mock_video_translation = Mock()
        long_segment = Mock()
        long_segment.start = 0.0
        long_segment.end = 5.0  # More than 4 seconds
        long_segment.translation = "Long translation"
        long_segment.source_file = str(tmp_path / "source.wav")
        long_segment.speaker = "SPEAKER_00"
        
        mock_video_translation.translated_texts = [long_segment]
        
        # Create source file
        Path(long_segment.source_file).touch()
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Verify add_pauses was called for long segment
        mock_add_pauses.assert_called_once_with(long_segment.source_file)
        # Verify merge_speaker_files was NOT called
        mock_merge.assert_not_called()
    
    def test_tts_pipeline_existing_files(self, mock_merge, mock_add_pauses, mock_tts_class, mock_file_repository, tmp_path):
        """Test TTS pipeline with existing output files."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create mock video translation
        mock_video_translation = Mock()
        segment = Mock()
        segment.start = 0.0
        segment.end = 2.0
        segment.translation = "Translation"
        segment.source_file = str(tmp_path / "source.wav")
        segment.speaker = "SPEAKER_00"
        
        mock_video_translation.translated_texts = [segment]
        
        # Create source file and output file (to simulate existing file)
        Path(segment.source_file).touch()
        output_file = tmp_path / "0.0_2.0.wav"
        output_file.touch()
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Verify TTS was NOT called since file already exists
        mock_tts_instance.tts_to_file.assert_not_called()
        assert segment.generated_file == str(output_file)
    
    @pytest.mark.parametrize("language", ["en", "es", "fr"])
    def test_tts_pipeline_language_mapping(self, mock_merge, mock_add_pauses, mock_tts_class, language, mock_file_repository, tmp_path):
        """Test TTS pipeline with different language mappings."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        with patch('src.ml.text_to_speech_service.tts_xtts_client.map_language_to_code') as mock_map:
            mock_map.return_value = language
            
            mock_video_translation = Mock()
            segment = Mock()
            segment.start = 0.0
            segment.end = 2.0
            segment.translation = "Translation"
            segment.source_file = str(tmp_path / "source.wav")
            segment.speaker = "SPEAKER_00"
            
            mock_video_translation.translated_texts = [segment]
            Path(segment.source_file).touch()
            
            client = XTTSClient(file_repository=mock_file_repository)
            
            client.tts_pipeline(mock_video_translation, str(tmp_path), language=language)
            
            # Verify language mapping was called
            mock_map.assert_called_once_with(language, "whisper")


class TestTTSEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_model_loading_failure(self, mock_tts_class, mock_file_repository):
        """Test handling of model loading failure."""
        mock_tts_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            XTTSClient(file_repository=mock_file_repository)
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_generate_audio_missing_source(self, mock_tts_class, mock_file_repository, sample_text, tmp_path):
        """Test audio generation with missing source file."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_instance.tts_to_file.side_effect = Exception("Source file not found")
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        non_existent_source = str(tmp_path / "non_existent.wav")
        save_path = str(tmp_path / "output.wav")
        
        with pytest.raises(Exception):
            client.generate_audio(
                text=sample_text,
                source_audio_file=non_existent_source,
                source_text="",
                save_path=save_path,
                language="en"
            )
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_empty_text_generation(self, mock_tts_class, mock_file_repository, tmp_path):
        """Test audio generation with empty text."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        source_audio = tmp_path / "source.wav"
        source_audio.touch()
        save_path = tmp_path / "output.wav"
        
        client.generate_audio(
            text="",  # Empty text
            source_audio_file=str(source_audio),
            source_text="",
            save_path=str(save_path),
            language="en"
        )
        
        # Should still call TTS (might generate silence or handle gracefully)
        mock_tts_instance.tts_to_file.assert_called_once()
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_device_switching(self, mock_tts_class, mock_file_repository):
        """Test device switching functionality."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Test CUDA device
        client_cuda = XTTSClient(file_repository=mock_file_repository, device="cuda")
        assert client_cuda.device == "cuda"
        
        # Test CPU device
        client_cpu = XTTSClient(file_repository=mock_file_repository, device="cpu")
        assert client_cpu.device == "cpu"
        
        # Verify model.to() was called with correct device
        assert mock_tts_instance.to.call_count >= 2
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_invalid_language_code(self, mock_tts_class, mock_file_repository, sample_text, tmp_path):
        """Test handling of invalid language codes."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_instance.tts_to_file.side_effect = Exception("Unsupported language")
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        source_audio = tmp_path / "source.wav"
        source_audio.touch()
        save_path = tmp_path / "output.wav"
        
        with pytest.raises(Exception):
            client.generate_audio(
                text=sample_text,
                source_audio_file=str(source_audio),
                source_text="",
                save_path=str(save_path),
                language="invalid_lang"
            )
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_tts_pipeline_empty_segments(self, mock_tts_class, mock_file_repository, tmp_path):
        """Test TTS pipeline with empty segments list."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        mock_video_translation = Mock()
        mock_video_translation.translated_texts = []  # Empty list
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        assert result == mock_video_translation
        # Verify no TTS calls were made
        mock_tts_instance.tts_to_file.assert_not_called()
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_tts_pipeline_missing_attributes(self, mock_tts_class, mock_file_repository, tmp_path):
        """Test TTS pipeline with segments missing required attributes."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        mock_video_translation = Mock()
        incomplete_segment = Mock()
        # Missing required attributes like start, end, translation, etc.
        
        mock_video_translation.translated_texts = [incomplete_segment]
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        with pytest.raises(AttributeError):
            client.tts_pipeline(mock_video_translation, str(tmp_path))


class TestTTSIntegration:
    """Integration tests for TTS workflow."""
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    @patch('src.ml.text_to_speech_service.tts_xtts_client.add_pauses')
    @patch('src.ml.text_to_speech_service.tts_xtts_client.merge_speaker_files')
    def test_full_tts_workflow(self, mock_merge, mock_add_pauses, mock_tts_class, mock_file_repository, sample_segments, tmp_path):
        """Test complete TTS workflow with multiple segments and speakers."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create comprehensive mock video translation
        mock_video_translation = Mock()
        segments = []
        
        for i in range(3):
            segment = Mock()
            segment.start = float(i * 2)
            segment.end = float(i * 2 + 1.5)  # Mixed short/long segments
            segment.translation = f"Translation segment {i}"
            segment.source_file = str(tmp_path / f"source_{i}.wav")
            segment.speaker = f"SPEAKER_0{i % 2}"  # Alternate speakers
            
            # Create source files
            Path(segment.source_file).touch()
            segments.append(segment)
        
        mock_video_translation.translated_texts = segments
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path), language="en")
        
        # Verify complete workflow
        assert result == mock_video_translation
        assert mock_tts_instance.tts_to_file.call_count == len(segments)
        
        # Verify all segments have generated files assigned
        for segment in segments:
            assert hasattr(segment, 'generated_file')
            assert segment.generated_file is not None
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_context_manager_integration(self, mock_tts_class, mock_file_repository, sample_text, tmp_path):
        """Test XTTSClient integration with context manager."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        source_audio = tmp_path / "source.wav"
        source_audio.touch()
        save_path = tmp_path / "output.wav"
        
        # Test with context manager
        with XTTSClient(file_repository=mock_file_repository) as client:
            client.generate_audio(
                text=sample_text,
                source_audio_file=str(source_audio),
                source_text="",
                save_path=str(save_path),
                language="en"
            )
            
            # Model should be available inside context
            assert client.model is not None
            mock_tts_instance.tts_to_file.assert_called_once()
        
        # Model should be cleared after context exit
        assert client.model is None
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    @patch('src.ml.text_to_speech_service.tts_xtts_client.map_language_to_code')
    def test_language_processing_integration(self, mock_map, mock_tts_class, mock_file_repository, tmp_path):
        """Test integration of language processing and mapping."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Test language mapping integration
        test_languages = ["English", "Spanish", "French"]
        mapped_codes = ["en", "es", "fr"]
        
        for lang, code in zip(test_languages, mapped_codes):
            mock_map.return_value = code
            
            mock_video_translation = Mock()
            segment = Mock()
            segment.start = 0.0
            segment.end = 2.0
            segment.translation = f"Text in {lang}"
            segment.source_file = str(tmp_path / "source.wav")
            segment.speaker = "SPEAKER_00"
            
            mock_video_translation.translated_texts = [segment]
            Path(segment.source_file).touch()
            
            client = XTTSClient(file_repository=mock_file_repository)
            client.tts_pipeline(mock_video_translation, str(tmp_path), language=lang)
            
            # Verify language was mapped and used correctly
            mock_map.assert_called_with(lang, "whisper")
            
            # Reset mock for next iteration
            mock_map.reset_mock()
            mock_tts_instance.reset_mock()


class TestTTSPerformance:
    """Performance and resource management tests."""
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_memory_cleanup(self, mock_tts_class, mock_file_repository):
        """Test that resources are properly cleaned up."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        # Simulate resource cleanup
        with client:
            assert client.model is not None
        
        # Verify cleanup occurred
        assert client.model is None
    
    @patch('src.ml.text_to_speech_service.tts_xtts_client.TTS')
    def test_batch_processing_efficiency(self, mock_tts_class, mock_file_repository, tmp_path):
        """Test efficient batch processing of multiple segments."""
        mock_tts_instance = Mock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_class.return_value = mock_tts_instance
        
        # Create large batch of segments
        mock_video_translation = Mock()
        segments = []
        
        for i in range(10):
            segment = Mock()
            segment.start = float(i)
            segment.end = float(i + 1)
            segment.translation = f"Batch segment {i}"
            segment.source_file = str(tmp_path / f"source_{i}.wav")
            segment.speaker = "SPEAKER_00"
            
            Path(segment.source_file).touch()
            segments.append(segment)
        
        mock_video_translation.translated_texts = segments
        
        client = XTTSClient(file_repository=mock_file_repository)
        
        result = client.tts_pipeline(mock_video_translation, str(tmp_path))
        
        # Verify all segments were processed
        assert mock_tts_instance.tts_to_file.call_count == len(segments)
        assert result == mock_video_translation