#!/bin/bash


# # Run the Python script with the specified arguments
python inference-cli.py \
    --model F5-TTS \
    --gen_text "Why? Why? What and how? Well, why and why are fairly clear here, but why sales? You can easily figure out why they happen. But what are the sales goals? How are sales carried out? What is the sales strategy? A big question. What are the goals of procurement?" \
    --out_file "generated_speech.wav" \
    --speed 1.0