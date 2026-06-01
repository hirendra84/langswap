import subprocess
import runpod
from main import process_translation, process_update_translation

# Set GPU compute mode at startup
try:
    subprocess.run(['nvidia-smi', '-c', '0'], check=True)
    print("Successfully set GPU compute mode to DEFAULT")
except subprocess.CalledProcessError as e:
    print(f"Warning: Failed to set GPU compute mode: {e}")
except FileNotFoundError:
    print("Warning: nvidia-smi not found, skipping GPU compute mode configuration")

def handler(job):
    """RunPod serverless handler that uses main.py functionality"""
    input = job['input']
    show_progress = input.get("show_progress", False)
    
    # Create a progress callback function specific to runpod
    def progress_callback(message):
        if show_progress:
            runpod.serverless.progress_update(job, message)
    
    # Call the shared process_translation function
    if "update_request" in input:
        process_update_translation(input, progress_callback)
    return process_translation(input, progress_callback)

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
