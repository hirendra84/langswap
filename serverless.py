import sys
import runpod
from main import process_translation, test_video_translation_local

def handler(job):
    """RunPod serverless handler that uses main.py functionality"""
    input = job['input']
    show_progress = input.get("show_progress", False)
    
    # Create a progress callback function specific to runpod
    def progress_callback(message):
        if show_progress:
            runpod.serverless.progress_update(job, message)
    
    # Call the shared process_translation function
    return process_translation(input, progress_callback)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
