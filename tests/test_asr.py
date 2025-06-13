"""
Tests for ASR (Automatic Speech Recognition) components.
Following the albumentations testing pattern with fixtures and parametrization.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.speech_to_text_service.asr_client import ASRX, Segment, Output, TranscriptionData


class TestSegment:
    """Test Segment data class."""
    
    def test_segment_creation(self):
        """Test basic segment creation."""
        segment = Segment(
            start=0.0,
            end=1.0,
            text="test text",
            words=[],
            speaker="SPEAKER_00"
        )
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "test text"
        assert segment.speaker == "SPEAKER_00"
    
    @pytest.mark.parametrize("start,end,text", [
        (0.0, 1.0, "short"),
        (1.5, 3.2, "longer test sentence"),
        (10.0, 15.5, ""),
    ])
    def test_segment_parametrized(self, start, end, text):
        """Test segment creation with different parameters."""
        segment = Segment(start=start, end=end, text=text, words=[])
        assert segment.start == start
        assert segment.end == end
        assert segment.text == text


class TestOutput:
    """Test Output data class."""
    
    def test_output_creation(self):
        """Test basic output creation."""
        segments = [
            Segment(start=0.0, end=1.0, text="test", words=[])
        ]
        output = Output(
            detected_language="en",
            device="cpu",
            model="whisper",
            transcription="test transcription",
            segments=segments
        )
        assert output.detected_language == "en"
        assert output.device == "cpu"
        assert len(output.segments) == 1


@patch('src.ml.speech_to_text_service.asr_client.whisperx')
class TestASRX:
    """Test ASRX class with mocked dependencies."""
    
    def test_init(self, mock_whisperx, mock_device):
        """Test ASRX initialization."""
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language="en")
            assert asr.device == mock_device
            assert asr.language == "en"
    
    def test_init_no_language(self, mock_whisperx, mock_device):
        """Test ASRX initialization without language."""
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language=None)
            assert asr.language is None
    
    @pytest.mark.parametrize("language", ["en", "es", "fr", "de"])
    def test_init_different_languages(self, mock_whisperx, language, mock_device):
        """Test ASRX initialization with different languages."""
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language=language)
            assert asr.language == language
    
    def test_load_models(self, mock_whisperx, mock_device):
        """Test model loading."""
        mock_whisperx.load_model.return_value = Mock()
        mock_whisperx.DiarizationPipeline.return_value = Mock()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.chdir'), \
             patch('pathlib.Path.cwd'), \
             patch('pathlib.Path.resolve'):
            asr = ASRX(device=mock_device, language="en")
            asr.load_models()
            
            mock_whisperx.load_model.assert_called_once()
            mock_whisperx.DiarizationPipeline.assert_called_once()
    
    def test_context_manager(self, mock_whisperx, mock_device):
        """Test ASRX as context manager."""
        with patch('os.path.exists', return_value=True):
            with ASRX(device=mock_device, language="en") as asr:
                assert asr.model is not None
            # After exit, models should be cleared
            assert asr.model is None
            assert asr.diarize_model is None
    
    def test_transcribe_basic(self, mock_whisperx, mock_device, sample_audio_file):
        """Test basic transcription functionality."""
        # Setup mocks
        mock_whisperx.load_audio.return_value = np.array([0.1, 0.2, 0.3])
        mock_whisperx.load_align_model.return_value = (Mock(), Mock())
        mock_whisperx.align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "test",
                    "words": [{"word": "test", "start": 0.0, "end": 1.0}]
                }
            ]
        }
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "test",
                    "words": [{"word": "test", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
                }
            ]
        }
        
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language="en")
            asr.model = Mock()
            asr.model.transcribe.return_value = {
                "segments": [],
                "language": "en"
            }
            asr.diarize_model = Mock()
            asr.diarize_model.return_value = {"segments": []}
            
            result = asr.transcribe(sample_audio_file)
            
            assert isinstance(result, Output)
            assert result.detected_language == "en"
            mock_whisperx.load_audio.assert_called_once_with(sample_audio_file)
    
    @pytest.mark.parametrize("num_speakers", [None, 2, 3, 5])
    def test_transcribe_with_speakers(self, mock_whisperx, num_speakers, mock_device, sample_audio_file):
        """Test transcription with different number of speakers."""
        # Setup basic mocks
        mock_whisperx.load_audio.return_value = np.array([0.1, 0.2, 0.3])
        mock_whisperx.load_align_model.return_value = (Mock(), Mock())
        mock_whisperx.align.return_value = {"segments": []}
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}
        
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language="en")
            asr.model = Mock()
            asr.model.transcribe.return_value = {"segments": [], "language": "en"}
            asr.diarize_model = Mock()
            asr.diarize_model.return_value = {"segments": []}
            
            result = asr.transcribe(sample_audio_file, num_speakers=num_speakers)
            
            # Verify diarization was called with correct parameters
            asr.diarize_model.assert_called_once()
            call_args = asr.diarize_model.call_args
            if num_speakers is not None:
                assert call_args[1]['num_speakers'] == num_speakers
    
    def test_get_cache_dir(self, mock_whisperx, mock_device):
        """Test cache directory retrieval."""
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language="en")
            cache_dir = asr.get_cache_dir()
            assert cache_dir is not None
            assert isinstance(cache_dir, (str, Path))


class TestTranscriptionDataStructures:
    """Test transcription data structures and conversions."""
    
    def test_transcription_data_creation(self):
        """Test TranscriptionData creation."""
        output = Output(
            detected_language="en",
            device="cpu",
            model="whisper",
            transcription="test",
            segments=[]
        )
        data = TranscriptionData(output=output)
        assert data.output == output
    
    def test_segment_with_speaker_assignment(self):
        """Test segment creation with speaker assignment."""
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"word": "world", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}
        ]
        segment = Segment(
            start=0.0,
            end=1.0,
            text="hello world",
            words=words,
            speaker="SPEAKER_00"
        )
        assert segment.speaker == "SPEAKER_00"
        assert len(segment.words) == 2
        assert all(word["speaker"] == "SPEAKER_00" for word in segment.words)


class TestASREdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('src.ml.speech_to_text_service.asr_client.whisperx')
    def test_missing_diarization_model(self, mock_whisperx, mock_device):
        """Test behavior when diarization model is missing."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                ASRX(device=mock_device, language="en")
    
    @patch('src.ml.speech_to_text_service.asr_client.whisperx')
    def test_empty_audio_file(self, mock_whisperx, mock_device, tmp_path):
        """Test handling of empty audio file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()
        
        mock_whisperx.load_audio.side_effect = Exception("Invalid audio file")
        
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device=mock_device, language="en")
            asr.model = Mock()
            asr.diarize_model = Mock()
            
            with pytest.raises(Exception):
                asr.transcribe(str(empty_file))
    
    @patch('src.ml.speech_to_text_service.asr_client.whisperx')
    def test_device_switching(self, mock_whisperx, mock_device):
        """Test device switching behavior."""
        with patch('os.path.exists', return_value=True):
            asr = ASRX(device="cuda", language="en")
            assert asr.device == "cuda"
            
            # Test with CPU fallback
            asr_cpu = ASRX(device="cpu", language="en")
            assert asr_cpu.device == "cpu" 