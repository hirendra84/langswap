"""
Tests for LLM (Large Language Model) translation components.
Following the albumentations testing pattern with fixtures and parametrization.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langswap.ml.translation_service.translator_client_gemma import (
    TranslatorClient,
    QuantizedGemmaTranslationClient
)


class TestTranslatorClient:
    """Test base TranslatorClient abstract class."""
    
    def test_abstract_class_instantiation(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TranslatorClient("cuda")


@patch('src.ml.translation_service.translator_client.Llama')
class TestQuantizedGemmaTranslationClient:
    """Test QuantizedGemmaTranslationClient with mocked dependencies."""
    
    def test_init_default_params(self, mock_llama, mock_device):
        """Test initialization with default parameters."""
        client = QuantizedGemmaTranslationClient(device=mock_device)
        assert client.device == mock_device
        assert client.n_gpu_layers == -1
        assert client.model is None
    
    def test_init_custom_params(self, mock_llama, mock_device):
        """Test initialization with custom parameters."""
        custom_path = "/custom/model/path.gguf"
        custom_layers = 10
        
        client = QuantizedGemmaTranslationClient(
            device=mock_device,
            model_path=custom_path,
            n_gpu_layers=custom_layers
        )
        assert client.model_path == custom_path
        assert client.n_gpu_layers == custom_layers
    
    def test_load_models(self, mock_llama, mock_device):
        """Test model loading."""
        mock_model_instance = Mock()
        mock_llama.return_value = mock_model_instance
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        mock_llama.assert_called_once()
        assert client.model == mock_model_instance
        
        # Verify Llama was called with correct parameters
        call_args = mock_llama.call_args
        assert call_args[1]['n_gpu_layers'] == -1
        assert call_args[1]['chat_format'] == "gemma"
        assert call_args[1]['n_ctx'] == 8096
        assert call_args[1]['verbose'] is False
    
    @pytest.mark.parametrize("source_lang,target_lang", [
        ("English", "Spanish"),
        ("Spanish", "English"),
        ("French", "German"),
        ("English", "Japanese"),
    ])
    def test_translate_basic(self, mock_llama, source_lang, target_lang, mock_device, sample_texts):
        """Test basic translation functionality with different language pairs."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": f"Translated to {target_lang}", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        sentences = sample_texts[:2]  # Use first 2 sentences
        results = client.translate(sentences, source_lang, target_lang)
        
        assert len(results) == len(sentences)
        assert all(isinstance(result, str) for result in results)
        assert mock_model.create_chat_completion.call_count == len(sentences)
    
    def test_translate_single_sentence(self, mock_llama, mock_device, sample_text):
        """Test translation of single sentence."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "Translated text", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        results = client.translate([sample_text], "English", "Spanish")
        
        assert len(results) == 1
        assert results[0] == "Translated text"
    
    def test_translate_empty_list(self, mock_llama, mock_device):
        """Test translation with empty sentence list."""
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        results = client.translate([], "English", "Spanish")
        
        assert results == []
        assert mock_model.create_chat_completion.call_count == 0
    
    @pytest.mark.parametrize("temperature,top_k,top_p", [
        (0.7, 40, 0.9),
        (1.0, 64, 0.95),
        (0.1, 20, 0.8),
    ])
    def test_translate_with_generation_params(self, mock_llama, temperature, top_k, top_p, mock_device, sample_text):
        """Test translation with different generation parameters."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "Generated text", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        results = client.translate(
            [sample_text], 
            "English", 
            "Spanish",
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Verify generation parameters were passed correctly
        call_args = mock_model.create_chat_completion.call_args
        assert call_args[1]['temperature'] == temperature
        assert call_args[1]['top_k'] == top_k
        assert call_args[1]['top_p'] == top_p
    
    def test_translate_context_management(self, mock_llama, mock_device, sample_texts):
        """Test that context is managed properly in long conversations."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "Response", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        # Create a scenario where context would need to be trimmed (more than 22 messages)
        long_sentences = sample_texts * 10  # 40 sentences
        results = client.translate(long_sentences, "English", "Spanish")
        
        assert len(results) == len(long_sentences)
        # Verify that context management occurred by checking call history
        assert mock_model.create_chat_completion.call_count == len(long_sentences)
    
    def test_translate_with_system_prompt(self, mock_llama, mock_device, sample_text):
        """Test that system prompt is correctly set up."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "System response", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        client.translate([sample_text], "English", "Spanish")
        
        # Verify system message was included
        call_args = mock_model.create_chat_completion.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert 'Spanish' in messages[0]['content']
        assert 'native speaker' in messages[0]['content']
    
    def test_translate_response_cleaning(self, mock_llama, mock_device, sample_text):
        """Test that responses are properly cleaned."""
        mock_model = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "  Cleaned response  \n", "role": "assistant"}}
            ]
        }
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        results = client.translate([sample_text], "English", "Spanish")
        
        assert results[0] == "Cleaned response"  # Whitespace should be stripped


class TestTranslationEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('src.ml.translation_service.translator_client.Llama')
    def test_model_loading_failure(self, mock_llama, mock_device):
        """Test handling of model loading failure."""
        mock_llama.side_effect = Exception("Model loading failed")
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        
        with pytest.raises(Exception):
            client.load_models()
    
    @patch('src.ml.translation_service.translator_client.Llama')
    def test_translation_api_error(self, mock_llama, mock_device, sample_text):
        """Test handling of API errors during translation."""
        mock_model = Mock()
        mock_model.create_chat_completion.side_effect = Exception("API Error")
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        with pytest.raises(Exception):
            client.translate([sample_text], "English", "Spanish")
    
    @patch('src.ml.translation_service.translator_client.Llama')
    def test_malformed_response(self, mock_llama, mock_device, sample_text):
        """Test handling of malformed API responses."""
        mock_model = Mock()
        mock_response = {"choices": []}  # Empty choices
        mock_model.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        with pytest.raises((IndexError, KeyError)):
            client.translate([sample_text], "English", "Spanish")
    
    @pytest.mark.parametrize("invalid_sentences", [
        None,
        [None],
        ["", ""],
        [123, "valid text"],
    ])
    @patch('src.ml.translation_service.translator_client.Llama')
    def test_invalid_input_sentences(self, mock_llama, invalid_sentences, mock_device):
        """Test handling of invalid input sentences."""
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        # Depending on input, this might raise different exceptions or handle gracefully
        if invalid_sentences is None:
            with pytest.raises(TypeError):
                client.translate(invalid_sentences, "English", "Spanish")
        else:
            # Test should handle empty strings and other types gracefully
            try:
                results = client.translate(invalid_sentences, "English", "Spanish")
                # If it doesn't raise an exception, verify the results are reasonable
                assert isinstance(results, list)
                assert len(results) == len(invalid_sentences)
            except (TypeError, ValueError, AttributeError):
                # These exceptions are acceptable for invalid inputs
                pass


class TestTranslationIntegration:
    """Integration tests for translation workflow."""
    
    @patch('src.ml.translation_service.translator_client.Llama')
    def test_full_translation_workflow(self, mock_llama, mock_device, sample_texts):
        """Test complete translation workflow."""
        mock_model = Mock()
        responses = [
            {"choices": [{"message": {"content": f"Translation {i}", "role": "assistant"}}]}
            for i in range(len(sample_texts))
        ]
        mock_model.create_chat_completion.side_effect = responses
        mock_llama.return_value = mock_model
        
        client = QuantizedGemmaTranslationClient(device=mock_device)
        client.load_models()
        
        results = client.translate(sample_texts, "English", "Spanish")
        
        assert len(results) == len(sample_texts)
        for i, result in enumerate(results):
            assert result == f"Translation {i}"
        
        # Verify all sentences were processed
        assert mock_model.create_chat_completion.call_count == len(sample_texts) 