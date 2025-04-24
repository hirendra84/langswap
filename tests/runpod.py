import os
import json
import requests
import time
from dotenv import load_dotenv
import boto3
from botocore.client import Config
import argparse
import sys
import logging
from datetime import datetime

# Import the local testing function from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import test_video_translation_local

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    return {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'runpod_api_key': os.getenv('RUNPOD_API_KEY')
    }

def get_s3_client(credentials):
    """Create and return an S3 client for Yandex Cloud"""
    return boto3.client(
        's3',
        aws_access_key_id=credentials['aws_access_key_id'],
        aws_secret_access_key=credentials['aws_secret_access_key'],
        endpoint_url='https://storage.yandexcloud.net',
        config=Config(signature_version='s3v4')
    )

def list_test_videos(s3_client, bucket_name='langswap-videos-dev', prefix='tests/'):
    """List all test videos in the specified S3 bucket path"""
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix
    )
    
    videos = []
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.mp4'):
                videos.append(obj['Key'])
    
    return videos

def generate_presigned_url(s3_client, bucket_name, object_key, expiration=2592000):
    """Generate a pre-signed URL for an S3 object with a long expiration time"""
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket_name,
            'Key': object_key
        },
        ExpiresIn=expiration
    )
    return url

def get_target_language_for_video(video_key):
    """Determine target language based on the source video filename"""
    # Extract source language from filename (assuming format like 'tests/english.mp4')
    source_language = os.path.basename(video_key).replace('.mp4', '').lower()
    
    # Map of source languages to target languages
    # Each source language video will be translated to a different target language
    language_map = {
        'arabic': 'english',
        'czech': 'spanish',
        'dutch': 'german',
        'english': 'french',
        'french': 'italian',
        'german': 'dutch',
        'hindi': 'russian',
        'hungarian': 'polish',
        'italian': 'portuguese',
        'japanese': 'korean',
        'korean': 'japanese',
        'polish': 'hungarian',
        'portuguese': 'turkish',
        'russian': 'hindi',
        'spanish': 'czech',
        'turkish': 'arabic'
    }
    
    # If source language is in our map, return its target, otherwise default to English
    return language_map.get(source_language, 'english')

def submit_translation_job(video_url, runpod_api_key, video_key, endpoint_id='imukd6fpsg4hk4'):
    """Submit a translation job to RunPod API"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {runpod_api_key}'
    }
    
    # Get target language based on the video filename
    target_language = get_target_language_for_video(video_key)
    
    # Prepare job input data (based on test_input.json)
    input_data = {
        "input": {
            "target_language": target_language,
            "tts_engine": "xtts",
            "watermark": True,
            "name": f"test_{os.path.basename(video_url.split('?')[0])}",
            "public_id": f"test_{int(time.time())}",
            "s3_video_url": video_url
        }
    }
    
    response = requests.post(
        f'https://api.runpod.ai/v2/{endpoint_id}/run',
        headers=headers,
        json=input_data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error submitting job: {response.status_code} - {response.text}")
        return None

def check_job_status(job_id, runpod_api_key, endpoint_id='imukd6fpsg4hk4'):
    """Check the status of a RunPod job"""
    headers = {
        'Authorization': f'Bearer {runpod_api_key}'
    }
    
    response = requests.get(
        f'https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}',
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error checking job status: {response.status_code} - {response.text}")
        return None

def setup_video_logger(video_key):
    """Create a logger that writes to a file named after the video being processed"""
    # Extract the video filename without the path and extension
    video_filename = os.path.basename(video_key).replace('.mp4', '')
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{video_filename}_{timestamp}.txt")
    
    # Configure logger
    logger = logging.getLogger(video_filename)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Starting log for video: {video_key}")
    return logger

def run_translation_tests():
    """Main function to run the translation tests"""
    # Load environment variables
    credentials = load_environment()
    
    # Create S3 client
    s3_client = get_s3_client(credentials)
    
    # List all test videos
    videos = list_test_videos(s3_client)
    print(f"Found {len(videos)} test videos")
    
    # Track submitted jobs
    submitted_jobs = []
    
    # Process each video
    for video_key in videos:
        # Set up logging for this video
        logger = setup_video_logger(video_key)
        
        # Generate pre-signed URL
        video_url = generate_presigned_url(s3_client, 'langswap-videos-dev', video_key)
        target_language = get_target_language_for_video(video_key)
        logger.info(f"Processing video: {video_key} with target language: {target_language}")
        
        # Submit translation job
        job_response = submit_translation_job(video_url, credentials['runpod_api_key'], video_key)
        
        if job_response and 'id' in job_response:
            job_id = job_response['id']
            submitted_jobs.append({
                'video_key': video_key,
                'job_id': job_id,
                'status': 'submitted',
                'logger': logger
            })
            logger.info(f"Job submitted with ID: {job_id}")
        else:
            logger.error(f"Failed to submit job")
    
    # Monitor job status (optional - can be expanded to wait for completion)
    print(f"\nSubmitted {len(submitted_jobs)} jobs for translation")
    
    # Return the list of submitted jobs for further tracking
    return submitted_jobs

def wait_for_job_completion(submitted_jobs, runpod_api_key, check_interval=60, timeout=3600):
    """Wait for all jobs to complete with a timeout"""
    start_time = time.time()
    completed_jobs = 0
    
    while completed_jobs < len(submitted_jobs) and (time.time() - start_time) < timeout:
        for job in submitted_jobs:
            if job['status'] in ['completed', 'failed']:
                continue
                
            logger = job['logger']
            status_response = check_job_status(job['job_id'], runpod_api_key)
            
            if status_response:
                current_status = status_response.get('status', '')
                
                if current_status == 'COMPLETED':
                    job['status'] = 'completed'
                    job['result'] = status_response.get('output', {})
                    completed_jobs += 1
                    logger.info(f"Job {job['job_id']} completed successfully")
                    logger.info(f"Output: {json.dumps(job['result'], indent=2)}")
                    
                elif current_status in ['FAILED', 'CANCELLED']:
                    job['status'] = 'failed'
                    job['error'] = status_response.get('error', 'Unknown error')
                    completed_jobs += 1
                    logger.error(f"Job failed: {job['error']}")
                else:
                    logger.info(f"Current status: {current_status}")
        
        # If not all jobs completed, wait before checking again
        if completed_jobs < len(submitted_jobs):
            print(f"Waiting for {len(submitted_jobs) - completed_jobs} jobs to complete...")
            time.sleep(check_interval)
    
    # Check for timeout
    if (time.time() - start_time) >= timeout:
        print("Timeout reached while waiting for jobs to complete")
    
    # Summarize results
    successful = sum(1 for job in submitted_jobs if job['status'] == 'completed')
    failed = sum(1 for job in submitted_jobs if job['status'] == 'failed')
    pending = len(submitted_jobs) - successful - failed
    
    print(f"\nTranslation Test Results:")
    print(f"Total jobs: {len(submitted_jobs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Pending: {pending}")
    
    # Return True if all jobs completed successfully
    return successful == len(submitted_jobs)

def run_local_translation_tests():
    """Run the translation tests locally without using RunPod"""
    # Load environment variables
    credentials = load_environment()
    
    # Create S3 client
    s3_client = get_s3_client(credentials)
    
    # List all test videos
    videos = list_test_videos(s3_client)
    print(f"Found {len(videos)} test videos")
    
    # Process each video
    results = []
    for video_key in videos:
        # Set up logging for this video
        logger = setup_video_logger(video_key)
        
        # Generate pre-signed URL
        video_url = generate_presigned_url(s3_client, 'langswap-videos-dev', video_key)
        target_language = get_target_language_for_video(video_key)
        logger.info(f"Processing video locally: {video_key} with target language: {target_language}")
        
        # Create a test input for the local translation pipeline
        test_input = {
            "input": {
                "target_language": target_language,
                "tts_engine": "xtts",
                "watermark": True,
                "name": f"test_{os.path.basename(video_key)}",
                "public_id": f"test_local_{int(time.time())}",
                "s3_video_url": video_url
            }
        }
        logger.info(f"Test input: {json.dumps(test_input, indent=2)}")
        
        # Save the test input as a temporary file
        temp_input_file = f"temp_input_{int(time.time())}.json"
        with open(temp_input_file, "w") as f:
            json.dump(test_input, f)
        
        try:
            # Run the local translation test
            logger.info(f"Starting local translation")
            
            # Capture stdout/stderr for logging
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                test_video_translation_local(temp_input_file)
            
            # Log captured output
            logger.info("Standard output:\n" + stdout_capture.getvalue())
            if stderr_capture.getvalue():
                logger.warning("Standard error:\n" + stderr_capture.getvalue())
            
            logger.info("Translation completed successfully")
            results.append({
                'video_key': video_key,
                'status': 'completed'
            })
        except Exception as e:
            logger.error(f"Error processing: {str(e)}", exc_info=True)
            results.append({
                'video_key': video_key,
                'status': 'failed',
                'error': str(e)
            })
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_input_file):
                os.remove(temp_input_file)
                logger.info(f"Removed temporary input file: {temp_input_file}")
    
    # Summarize results
    successful = sum(1 for job in results if job['status'] == 'completed')
    failed = sum(1 for job in results if job['status'] == 'failed')
    
    print(f"\nLocal Translation Test Results:")
    print(f"Total videos: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return results

if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Run video translation tests')
    parser.add_argument('--local', action='store_true', help='Run tests locally without RunPod')
    args = parser.parse_args()
    
    if args.local:
        print("Running local translation tests...")
        results = run_local_translation_tests()
    else:
        print("Running RunPod translation tests...")
        submitted_jobs = run_translation_tests()
        # Wait for job completion
        success = wait_for_job_completion(submitted_jobs, load_environment()['runpod_api_key'])
        print(f"All tests passed: {success}")
