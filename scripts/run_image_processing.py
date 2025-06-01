import argparse
import requests
import time
import os
from pathlib import Path # Added for path manipulation

# Define the base URL of the API
API_BASE_URL = "http://localhost:8000/api/v1"  # Assuming the API runs on localhost:8000
POLL_INTERVAL = 5  # seconds

def main():
    epilog_text = """
Example usage:
  python scripts/run_image_processing.py /path/to/your/image.jpg "A vibrant floral pattern" --output_dir ./processed_images
  python scripts/run_image_processing.py input.png "Abstract geometric design" --negative_prompt "blurry, noisy" --no_upscale
"""
    parser = argparse.ArgumentParser(
        description="Process an image using the Style Transfer API.",
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("style_prompt", type=str, help="Style prompt for the image processing.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for the image processing.")

    pgroup_enhance = parser.add_mutually_exclusive_group()
    pgroup_enhance.add_argument("--enhance_quality", dest='enhance_quality', action='store_true', help="Enhance quality.")
    pgroup_enhance.add_argument("--no_enhance_quality", dest='enhance_quality', action='store_false', help="Do not enhance quality.")

    pgroup_remove_bg = parser.add_mutually_exclusive_group()
    pgroup_remove_bg.add_argument("--remove_background", dest='remove_background', action='store_true', help="Remove background.")
    pgroup_remove_bg.add_argument("--no_remove_background", dest='remove_background', action='store_false', help="Do not remove background.")

    pgroup_upscale = parser.add_mutually_exclusive_group()
    pgroup_upscale.add_argument("--upscale", dest='upscale', action='store_true', help="Upscale image.")
    pgroup_upscale.add_argument("--no_upscale", dest='upscale', action='store_false', help="Do not upscale image.")

    parser.set_defaults(enhance_quality=True, remove_background=False, upscale=True)

    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the processed image (default: current directory).")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    print(f"Processing image: {args.image_path}")
    print(f"Style prompt: {args.style_prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Enhance quality: {args.enhance_quality}")
    print(f"Remove background: {args.remove_background}")
    print(f"Upscale: {args.upscale}")

    process_url = f"{API_BASE_URL}/process-garment"

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {'file': (os.path.basename(args.image_path), open(args.image_path, 'rb'))}
    data = {
        "style_prompt": args.style_prompt,
        "negative_prompt": args.negative_prompt,
        "enhance_quality": str(args.enhance_quality).lower(),
        "remove_background": str(args.remove_background).lower(),
        "upscale": str(args.upscale).lower()
    }

    try:
        print("Submitting image for processing...")
        response = requests.post(process_url, files=files, data=data)
        response.raise_for_status()

        response_data = response.json()
        task_id = response_data.get("task_id")

        if task_id:
            print(f"Image processing task started. Task ID: {task_id}")
            print("Polling for status updates...")

            status_url = f"{API_BASE_URL}/status/{task_id}"

            while True:
                try:
                    status_response = requests.get(status_url)
                    status_response.raise_for_status()
                    status_data = status_response.json()

                    current_status = status_data.get("status")
                    progress = status_data.get("progress", 0)
                    message = status_data.get("message", "")

                    print(f"Status: {current_status} | Progress: {progress}% | Message: {message}")

                    if current_status == "completed":
                        print("Processing completed successfully.")
                        download_url = f"{API_BASE_URL}/download/{task_id}"
                        print(f"Downloading result from {download_url}...")

                        try:
                            download_response = requests.get(download_url, stream=True)
                            download_response.raise_for_status()

                            # Construct output filename, e.g., original_filename_processed.jpg
                            original_filename_stem = Path(args.image_path).stem
                            # Attempt to get content type for extension, default to .jpg
                            content_type = download_response.headers.get('Content-Type', 'image/jpeg') # Default to jpeg
                            extension = ".jpg" # Default
                            if 'image/png' in content_type:
                                extension = ".png"
                            elif 'image/webp' in content_type:
                                extension = ".webp"
                            elif 'image/jpeg' in content_type:
                                extension = ".jpg"

                            output_filename = f"{original_filename_stem}_{task_id}_processed{extension}"
                            save_path = output_path / output_filename

                            with open(save_path, 'wb') as f:
                                for chunk in download_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"Processed image saved to: {save_path.resolve()}")

                        except requests.exceptions.RequestException as e_download:
                            print(f"Error downloading result: {e_download}")
                        except Exception as e_save:
                            print(f"Error saving downloaded image: {e_save}")
                        break
                    elif current_status == "failed":
                        error_details = status_data.get("error", "No error details provided.")
                        print(f"Processing failed. Error: {error_details}")
                        break
                    elif current_status in ["pending", "processing", "queued"]:
                        time.sleep(POLL_INTERVAL)
                    else:
                        print(f"Unknown status received: {current_status}. Stopping.")
                        break

                except requests.exceptions.RequestException as e_status:
                    print(f"Error polling status: {e_status}. Retrying in {POLL_INTERVAL} seconds...")
                    time.sleep(POLL_INTERVAL)
                except Exception as e_status_parse:
                    print(f"Error parsing status response: {e_status_parse}. Retrying in {POLL_INTERVAL} seconds...")
                    time.sleep(POLL_INTERVAL)
        else:
            print("Error: Could not get task ID from API response.")
            print(response_data)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Is it running?")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} {e.response.reason}")
        try:
            error_detail = e.response.json().get('detail')
            if error_detail:
                print(f"Detail: {error_detail}")
            else:
                print(f"Response body: {e.response.text}")
        except ValueError: # If response is not JSON
            print(f"Response body: {e.response.text}")
    except requests.exceptions.RequestException as e: # Catch other request-related errors
        print(f"Error during API request: {e}")
        if e.response is not None:
            print(f"API Response: {e.response.text}")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'file' in files and files['file'] is not None and hasattr(files['file'][1], 'closed') and not files['file'][1].closed:
            files['file'][1].close()

if __name__ == "__main__":
    main()
