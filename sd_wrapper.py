from flask import Flask, request, jsonify, send_file, stream_with_context, Response
from urllib.parse import unquote
import os
import hashlib
import requests
import logging
import base64
import threading
import time
import re
from datetime import datetime
from queue import PriorityQueue, Empty
import sys

# --- Configure Logging to suppress Werkzeug's default access logs ---
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)
# --- End Werkzeug logging configuration ---

app = Flask(__name__)

# Define the image directory and the cleanup interval (in seconds) and size thresholds
MODEL = "Unknown"
IMAGE_DIR = "images"
CLEANUP_INTERVAL = 3600
MAX_CACHE_SIZE = 1 * 1024 * 1024 * 1024
DELETE_SIZE = 250 * 1024 * 1024
HR_CFG = 1

# --- Global variables for debounced error logging ---
LAST_LOGGED_ERROR_TYPE = None
error_log_lock = threading.Lock()

# --- Request Queue and Worker Thread Globals ---
requests_queue = PriorityQueue()
processing_active_event = threading.Event()

# SLA for tasks: 5 minutes (300 seconds)
TASK_SLA_SECONDS = 300
SLA_PRIORITY_BOOST = -1000

# --- End of new globals ---

# --- Helper function for logging once by type (thread-safe) ---
def log_once_by_type(log_function, error_type_key, message):
    global LAST_LOGGED_ERROR_TYPE, error_log_lock
    with error_log_lock:
        if LAST_LOGGED_ERROR_TYPE != error_type_key:
            log_function(message)
            LAST_LOGGED_ERROR_TYPE = error_type_key

# --- Helper functions (unchanged) ---
def get_scale(style: str) -> float:
    models = { "Gibli": 1.5, "Flux": 1.5, "Film-Noir": 1.5, "3D-Disney": 1.5, "Comic": 1.5, "2D-Disney": 1.5, "sai-anime": 1.5 }
    return models.get(style, 2)
def get_noise(style: str) -> float:
    models = { "Gibli": 0.3, "Flux": 0.3, "Film-Noir": 0.3, "3D-Disney": 0.3, "Comic": 0.3, "2D-Disney": 0.3, "sai-anime": 0.3 }
    return models.get(style, 0.5)
def get_override(style: str) -> str:
    models = { "Photo": "Flux", "": "Flux"}
    return models.get(style, style)
def get_styles(style: str) -> str:
    models = { "Test": "Anime" }
    models = { "Flux": "Photo" }
    return models.get(style, style)
def get_res_x(style: str) -> float:
    models = { "Gibli": 1.28, "Flux": 1.28, "Film-Noir": 1.28, "3D-Disney": 1.28, "Comic": 1.28, "2D-Disney": 1.28, "sai-anime": 1.28 }
    return models.get(style, 1)
def get_res_y(style: str) -> float:
    models = { "Gibli": 1.2, "Flux": 1.2, "Film-Noir": 1.2, "3D-Disney": 1.2, "Comic": 1.2, "2D-Disney": 1.2, "sai-anime": 1.2 }
    return models.get(style, 1)
def get_hr(style: str) -> bool:
    models = { "2D-Disney": True, "Anime": True, "sai-anime": True, "Comic": True, "Photo": True, "sai-photographic": True, "Film-Noir": True, "3D-Disney": True, "Pixar": True, "Flux": True }
    return models.get(style, True)
def get_cfg(style: str) -> int:
    models = { "Gibli": 1.1, "2D-Disney": 1.1, "Anime": 12, "sai-anime": 1.1, "Comic": 1.1, "Photo": 12, "sai-photographic": 12, "Film-Noir": 1.1, "3D-Disney": 1.1, "Pixar": 12, "Flux": 1.1 }
    return models.get(style, 12)
def get_steps(style: str) -> int:
    models = { "Gibli": 22, "2D-Disney": 22, "Anime": 80, "sai-anime": 22, "Comic": 22, "Photo": 80, "sai-photographic": 80, "Film-Noir": 22, "3D-Disney": 22, "Pixar": 60, "Flux": 22 }
    return models.get(style, 80)
def get_hr_steps(style: str) -> int:
    models = { "Gibli": 3, "2D-Disney": 3, "Anime": 16, "sai-anime": 3, "Comic": 3, "Photo": 16, "sai-photographic": 16, "Film-Noir": 3, "3D-Disney": 3, "Pixar": 12, "Flux": 3 }
    return models.get(style, 16)
def get_age_weight(style: str) -> float:
    models = { "Gibli": 1, "2D-Disney": 1, "Anime": 1.9, "sai-anime": 1, "Comic": 1, "Photo": 3, "sai-photographic": 3, "Film-Noir": 1, "3D-Disney": 1, "Pixar": 3, "Flux": 1 }
    return models.get(style, 1)
def get_model(style: str) -> str:
    models = { "Gibli": "IllustrationJuanerGhibli_v20.safetensors", "2D-Disney": "designPixar_v20.safetensors", "Anime": "sdxlYamersAnime_stageAnima.safetensors", "sai-anime": "designPixar_v20.safetensors", "Comic": "sexyToonsFeatPipa_20Flux.safetensors", "Photo": "realDream_sdxlLightning1.safetensors", "sai-photographic": "realDream_sdxlLightning1.safetensors", "Film-Noir": "copaxTimeless_xplus4GUFF.gguf", "3D-Disney": "designPixar_v20.safetensors", "Pixar": "wildcardxXLANIMATION_wildcardxXLANIMATION.safetensors", "Flux": "copaxTimeless_xplus4GUFF.gguf"}#"Flux": "smolmodeFluxPruned_q6KGGUF68gb.gguf" }
    return models.get(style, "realDream_sdxlLightning1.safetensors")
def get_upscaler(style: str) -> str:
    upscalers = { "Gibli": "4x_NMKD-Superscale-SP_178000_G", "2D-Disney": "None", "Anime": "None", "sai-anime": "None", "Comic": "4x-AnimeSharp", "Photo": "4x-UltraSharp", "sai-photographic": "4x-UltraSharp", "Film-Noir": "Lanczos", "3D-Disney": "4x_NMKD-Superscale-SP_178000_G", "Pixar": "4x-UltraSharp", "Flux": "4x_NMKD-Superscale-SP_178000_G" }
    return upscalers.get(style, "4x-UltraSharp")
def get_vae(style: str) -> str:
    vaes = {"Gibli": "None", "2D-Disney": "None", "Anime": "sdxl_vae.safetensors", "sai-anime": "None", "Comic": "None", "Photo": "None", "sai-photographic": "None", "Film-Noir": "None", "3D-Disney": "None", "Pixar": "None", "Flux": "None" }
    return vaes.get(style, "None")
def get_extras(style: str) -> list[str]:
    extras = { "Gibli": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "Flux": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "Film-Noir": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "3D-Disney": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "Comic": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "2D-Disney": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"], "sai-anime": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"] }
    #extras = { "Flux": ["t5xxl_fp8_e4m3fn.safetensors","clip_l.safetensors","flux.safetensors"] }
    return extras.get(style, ["None"])


# --- RequestTask Class ---
class RequestTask:
    def __init__(self, priority, timestamp, style, prompt, width, height, seed, image_path, requested_model, event):
        self.original_priority = priority
        self.priority = priority
        self.timestamp = timestamp
        self.style = style
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed
        self.image_path = image_path
        self.requested_model = requested_model
        self.event = event
        self.success = False
        self.last_sla_check_time = time.time()

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

# Function to re-prioritize stale tasks
def re_prioritize_stale_tasks(q: PriorityQueue):
    temp_list = []
    re_prioritized_count = 0
    current_time = time.time()

    while True:
        try:
            p, ts, task = q.get_nowait()
            if task.priority >= 0 and (current_time - task.timestamp) > TASK_SLA_SECONDS and \
               (current_time - task.last_sla_check_time) > 10:
                
                logging.warning(f"SLA breached for task (original priority {task.original_priority}, waited {int(current_time - task.timestamp)}s): {task.prompt[:50]}...")
                task.priority = SLA_PRIORITY_BOOST
                re_prioritized_count += 1
                task.last_sla_check_time = current_time
                temp_list.append((task.priority, task.timestamp, task))
            else:
                temp_list.append((p, ts, task))
        except Empty:
            break

    for item in temp_list:
        q.put(item)

    if re_prioritized_count > 0:
        logging.info(f"Re-prioritized {re_prioritized_count} tasks due to SLA breach.")
# --- Dedicated SD API Worker Thread Function ---
def api_worker():
    global MODEL, requests_queue, processing_active_event

    initial_idle_timeout = 60
    max_idle_timeout = 3600 * 24
    idle_timeout_duration = initial_idle_timeout
    last_idle_log_time = time.time()
    queue_check_interval = 1
    last_queue_check_time = time.time()

    while True:
        task = None
        try:
            current_time = time.time()

            if not requests_queue.empty():
                re_prioritize_stale_tasks(requests_queue)
                time.sleep(0.05)

            if (current_time - last_queue_check_time) >= queue_check_interval:
                task = requests_queue.get(timeout=0.1)
                priority, timestamp, current_task = task
                processing_active_event.set()

                idle_timeout_duration = initial_idle_timeout
                last_idle_log_time = time.time()
                last_queue_check_time = current_time

                logging.info(f"Worker processing request (priority {current_task.priority}, timestamp {current_task.timestamp}): {current_task.style}, {current_task.prompt}. Target Model: {current_task.requested_model}")
                # --- MODIFICATION START: Dynamic "Smart Payload" Construction ---
                is_model_switch = MODEL != current_task.requested_model
                # Force a load on the very first run or after an error state
                if MODEL == "Unknown" or MODEL == "None":
                    is_model_switch = True
                
                api_url = "http://localhost:7860/sdapi/v1/txt2img"
                # Base payload with parameters common to all requests
                payload = {
                    "prompt": f"{current_task.prompt.replace('AW', str(get_age_weight(get_override(current_task.style))))}",
                    "negative_prompt": f"warped, stretched, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra body parts, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, objects through body, username, watermark, signature, mutated body parts, deformed body features, bad eyes, camera flash, camera lighting, flash-photography, glare, extra feet, extra toes",
                    "styles": [get_styles(current_task.style)],
                                          
                                                                            
                                                                            
                                                                                                
                      
                                                                  
                    "cfg_scale": get_cfg(get_override(current_task.style)),
                    "steps": get_steps(get_override(current_task.style)),
                    "enable_hr": get_hr(get_override(current_task.style)),
                    "denoising_strength": get_noise(get_override(current_task.style)),
                    "hr_upscaler": get_upscaler(get_override(current_task.style)),
                    "hr_scale": get_scale(get_override(current_task.style)),
                    "hr_sampler_name": "Euler",
                    "hr_second_pass_steps": get_hr_steps(get_override(current_task.style)),
                    "hr_additional_modules": ["Use same choices"],
                    "hr_cfg": HR_CFG,
                    "sampler_index": "Euler",
                    "scheduler": "Beta",
                    "width": int(current_task.width)*get_res_x(get_override(current_task.style)),
                    "height": int(current_task.height)*get_res_y(get_override(current_task.style)),
                    "seed": int(current_task.seed)
                }

                # Dynamically create the override_settings
                override_payload = {
                    # Additional modules might change per style even if the base model is the same
                    "forge_additional_modules": get_extras(get_override(current_task.style))
                }

                if is_model_switch:
                    logging.info(f"Worker building 'Load & Generate' payload for model: {current_task.requested_model}")
                    # For a model switch, add the checkpoint and VAE to the override
                    override_payload["sd_model_checkpoint"] = current_task.requested_model
                    override_payload["sd_vae"] = get_vae(get_override(current_task.style))
                    # Make the settings "stick" after this request
                    payload["override_settings_restore_afterwards"] = False
                else:
                    logging.info(f"Worker building 'Generate-Only' payload for currently loaded model: {MODEL}")
                    # If model is already loaded, we don't send checkpoint/VAE.
                    # We can restore afterwards as we aren't changing the permanent state.
                    payload["override_settings_restore_afterwards"] = True

                                     
                                                                                                                                   
                                                                                                                              
                                    
                payload["override_settings"] = override_payload
                                                                           
                # --- MODIFICATION END ---

                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json"
                }

                try:
                    logging.info(f"Worker sending txt2img request...")
                    response = requests.post(api_url, json=payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    response_json = response.json()
                    image_data = response_json["images"][0]
                    image_data = base64.b64decode(image_data)
                    with open(current_task.image_path, "wb") as f:
                        f.write(image_data)
                    logging.info(f"Worker successfully generated and saved image: {current_task.image_path}")
                    current_task.success = True
                    # MODIFICATION: Only update the global MODEL state after a successful switch
                                          
                    if is_model_switch:
                                                    
                        MODEL = current_task.requested_model
                except requests.exceptions.RequestException as e:
                    error_msg = f"Worker error sending txt2img request to SD API: {e}"
                    log_once_by_type(logging.error, "WorkerSDApiRequestError", error_msg)
                    current_task.success = False
                                          
                                                                                  
                    MODEL = "None"
                                                 
                                 
                except (KeyError, ValueError, IndexError) as e:
                    error_msg = f"Worker error decoding image data or malformed response from txt2img: {e}"
                    log_once_by_type(logging.error, "WorkerImageDecodingError", error_msg)
                    current_task.success = False
                                          
                    MODEL = "None"
                                                 
                                 
                except Exception as e:
                    error_msg = f"Worker unexpected error during image generation (txt2img): {e}"
                    log_once_by_type(logging.error, "WorkerUnexpectedThreadError", error_msg)
                    current_task.success = False
                                          
                    MODEL = "None"

                                 
                        
                                                                                           
                current_task.event.set()

                temp_queue = []
                processed_same_model_count = 0

                if current_task.success:
                    while not requests_queue.empty():
                        try:
                            p, ts, next_task = requests_queue.get_nowait()
                            if next_task.requested_model == MODEL:
                                logging.info(f"Worker processing next batched request for current model ({MODEL}): {next_task.prompt}")
                                _execute_single_task_batched(next_task)
                                processed_same_model_count += 1
                            else:
                                temp_queue.append((p, ts, next_task))
                        except Empty:
                            break
                        except Exception as e:
                            logging.warning(f"Error peeking/draining queue for batching: {e}")
                            break

                for item in temp_queue:
                    requests_queue.put(item)

                if processed_same_model_count > 0:
                    logging.info(f"Worker processed {processed_same_model_count} additional requests for model {MODEL} in batch.")
        
            else:
                time_to_sleep = queue_check_interval - (current_time - last_queue_check_time)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep + 0.01)

        except Empty:
            processing_active_event.clear()

            current_time = time.time()
            if (current_time - last_idle_log_time) >= idle_timeout_duration:
                if idle_timeout_duration < 60:
                    idle_message = f"Queue is idle ({int(idle_timeout_duration)} seconds)."
                elif idle_timeout_duration < 3600:
                    idle_message = f"Queue is idle ({int(idle_timeout_duration / 60)} minutes)."
                else:
                    idle_message = f"Queue is idle ({int(idle_timeout_duration / 3600)} hours)."

                logging.info(idle_message)
                last_idle_log_time = current_time
                idle_timeout_duration = min(idle_timeout_duration * 2, max_idle_timeout)
            
            time.sleep(1)

        except Exception as e:
            logging.error(f"Worker thread encountered an unexpected error: {e}", exc_info=True)
            processing_active_event.clear()
            time.sleep(1)

# --- Helper function for batched execution (simplified) ---
def _execute_single_task_batched(task):
    global MODEL
    
    # --- MODIFICATION START: Use "Generate-Only" payload for batched requests ---
    api_url = "http://localhost:7860/sdapi/v1/txt2img"
    payload = {
        "prompt": f"{task.prompt.replace('AW', str(get_age_weight(get_override(task.style))))}",
        "negative_prompt": f"warped, stretched, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra body parts, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, objects through body, username, watermark, signature, mutated body parts, deformed body features, bad eyes, camera flash, camera lighting, flash-photography, glare, extra feet, extra toes",
        "styles": [get_styles(task.style)],
        # This function is only called when the model is already loaded.
        # Therefore, we use a minimal override payload that does NOT include the checkpoint.
        "override_settings": {
                                                        
                                                        
            "forge_additional_modules": get_extras(get_override(task.style))
        },
        "override_settings_restore_afterwards": True,
        "cfg_scale": get_cfg(get_override(task.style)),
        "steps": get_steps(get_override(task.style)),
        "enable_hr": get_hr(get_override(task.style)),
        "denoising_strength": get_noise(get_override(task.style)),
        "hr_upscaler": get_upscaler(get_override(task.style)),
        "hr_scale": get_scale(get_override(task.style)),
        "hr_sampler_name": "Euler",
        "hr_second_pass_steps": get_hr_steps(get_override(task.style)),
        "hr_additional_modules": ["Use same choices"],
        "hr_cfg": HR_CFG,
        "sampler_index": "Euler",
        "scheduler": "Beta",
        "width": int(task.width)*get_res_x(get_override(task.style)),
        "height": int(task.height)*get_res_y(get_override(task.style)),
        "seed": int(task.seed)
    }
    # --- MODIFICATION END ---
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        logging.info(f"Worker (batched) sending txt2img request for model {MODEL}...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        image_data = response_json["images"][0]
        image_data = base64.b64decode(image_data)
        with open(task.image_path, "wb") as f:
            f.write(image_data)
        logging.info(f"Worker (batched) successfully generated and saved image: {task.image_path}")
        task.success = True
    except requests.exceptions.RequestException as e:
        error_msg = f"Worker (batched) error sending txt2img request to SD API: {e}"
        log_once_by_type(logging.error, "WorkerSDApiRequestErrorBatch", error_msg)
        task.success = False
        MODEL = "None"
    except (KeyError, ValueError, IndexError) as e:
        error_msg = f"Worker (batched) error decoding image data or malformed response from txt2img: {e}"
        log_once_by_type(logging.error, "WorkerImageDecodingErrorBatch", error_msg)
        task.success = False
        MODEL = "None"
    except Exception as e:
        error_msg = f"Worker (batched) unexpected error during image generation (txt2img): {e}"
        log_once_by_type(logging.error, "WorkerUnexpectedThreadErrorBatch", error_msg)
        task.success = False
        MODEL = "None"
    finally:
        task.event.set()

# --- Caching and Cleanup functions (unchanged) ---
def get_stable_hash_filename(request_params_string: str) -> str:
    sha256_hash = hashlib.sha256(request_params_string.encode('utf-8')).hexdigest()
    return sha256_hash + ".png"

def get_directory_size(directory: str) -> int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def cleanup_old_images():
    logging.info("Starting cleanup of old images...")
    total_size = get_directory_size(IMAGE_DIR)
    if total_size > MAX_CACHE_SIZE:
        logging.info(f"Total size of images directory ({total_size} bytes) exceeds the maximum cache size ({MAX_CACHE_SIZE} bytes).")
        files = sorted(
            (os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)),
            key=os.path.getmtime
        )
        size_to_free = DELETE_SIZE
        size_freed = 0
        for file in files:
            file_size = os.path.getsize(file)
            os.remove(file)
            size_freed += file_size
            logging.info(f"Deleted {file} ({file_size} bytes).")
            if size_freed >= size_to_free:
                break
        logging.info(f"Freed {size_freed} bytes by deleting old images.")
    else:
        logging.info(f"Total size of images directory ({total_size} bytes) is within the maximum cache size ({MAX_CACHE_SIZE} bytes). No cleanup needed.")
    logging.info("Cleanup of old images completed.")

def start_cleanup_thread():
    def run_cleanup():
        while True:
            cleanup_old_images()
            time.sleep(CLEANUP_INTERVAL)

    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

# --- Main Flask route for image generation (unchanged) ---
@app.route('/p/<prompt>/')
def generate_image(prompt):
    global LAST_LOGGED_ERROR_TYPE, MODEL, requests_queue

    size_param = request.args.get('size')
    seed_param = request.args.get('seed')
    style_param = request.args.get('style')

    if not size_param:
        return jsonify({"error": "Missing 'size' parameter. Example: ?size=500x800"}), 400
    if not seed_param:
        return jsonify({"error": "Missing 'seed' parameter. Example: &seed=12345"}), 400
    if not style_param:
        return jsonify({"error": "Missing 'style' parameter. Example: &style=Photo"}), 400

    validation_errors = []
    width, height, seed = None, None, None

    try:
        width_str, height_str = size_param.split('x')
        width = int(width_str)
        height = int(height_str)
        if width > 1024 or height > 1024:
            validation_errors.append(f"Size too large: {width}x{height}. Max 1024x1024.")
        if width < 500 or height < 600:
            validation_errors.append(f"Size too small: {width}x{height}. Min 500x600.")
    except ValueError:
        validation_errors.append(f"Invalid 'size' format: '{size_param}'. Expected 'WxH'.")

    try:
        seed = int(seed_param)
    except ValueError:
        validation_errors.append(f"Invalid 'seed' format: '{seed_param}'. Must be an integer.")

    known_styles = {"Gibli", "2D-Disney", "Anime", "sai-anime", "Comic", "Photo", "sai-photographic",
                    "Film-Noir", "3D-Disney", "Pixar", "Flux", "Test"}
    if style_param not in known_styles:
        validation_errors.append(f"Invalid 'style': '{style_param}'. Refer to API documentation for supported styles.")

    if validation_errors:
        combined_error_message = "Refusing request due to multiple validation issues: " + "; ".join(validation_errors) + \
                                 f". Request for prompt '{prompt}'."
        log_once_by_type(logging.error, hashlib.sha256(combined_error_message.encode()).hexdigest(), combined_error_message)
        return jsonify({"error": combined_error_message}), 400

    with error_log_lock:
        LAST_LOGGED_ERROR_TYPE = None

    logging.info(f"Received valid request for image generation: {style_param}, {prompt}, {width}x{height}, {seed}")

    decoded_prompt = unquote(prompt)
    request_cache_key = f"{decoded_prompt}-{width}x{height}-{seed}-{style_param}"
    stable_hashed_filename = get_stable_hash_filename(request_cache_key)
    image_path = os.path.join(IMAGE_DIR, stable_hashed_filename)

    if os.path.exists(image_path):
        logging.info(f"Image already exists in cache at {image_path}. Returning cached image.")
        # Log queue size when a cached image is returned
        logging.info(f"Queue size after cached image served: {requests_queue.qsize()}")
        with error_log_lock:
            LAST_LOGGED_ERROR_TYPE = None
        return send_file(image_path, mimetype='image/png')
        
    current_requested_model = get_model(get_override(style_param))
    request_timestamp = time.time()
    
    if MODEL != "Unknown" and current_requested_model == MODEL:
        priority = 0
        logging.info(f"Assigning priority 0 to request for model {current_requested_model} (currently loaded).")
    else:
        priority = 1
        logging.info(f"Assigning priority 1 to request for model {current_requested_model} (requires switch from {MODEL}).")

    request_event = threading.Event()
    task = RequestTask(priority, request_timestamp, style_param, decoded_prompt, width, height, seed,
                       image_path, current_requested_model, request_event)

    requests_queue.put((priority, request_timestamp, task))
    logging.info(f"Request added to queue (priority {priority}, timestamp {request_timestamp}). Queue size: {requests_queue.qsize()}")

    GENERATION_TIMEOUT = 300
    task_start_wait = time.time()

    while not task.event.is_set():
        if (time.time() - task_start_wait) >= GENERATION_TIMEOUT:
            error_msg = f"Timeout waiting for image generation from queue for prompt: {prompt}"
            log_once_by_type(logging.error, "QueueImageGenerationTimeout", error_msg)
            logging.info(f"Queue size after timeout for prompt '{prompt}': {requests_queue.qsize()}") # Log on timeout
            return jsonify({"error": "Timeout waiting for image generation"}), 500
        time.sleep(5)

    if task.success and os.path.exists(image_path):
        with error_log_lock:
            LAST_LOGGED_ERROR_TYPE = None
        def generate():
            logging.info("Streaming response to client from queue processing...")
                
            with open(image_path, "rb") as f:
                yield from f
        
        response = Response(stream_with_context(generate()), mimetype="image/png")
        # Log queue size when a generated image is successfully sent
        logging.info(f"Queue size after generated image served: {requests_queue.qsize()}")
        return response
        
    else:
        error_msg = f"Failed to generate image from queue for prompt: {prompt}. (Task indicated failure or file not found)"
        log_once_by_type(logging.error, "QueueImageGenerationFailed", error_msg)
        # Log queue size when a generation fails
        logging.info(f"Queue size after generation failure for prompt '{prompt}': {requests_queue.qsize()}")
        return jsonify({"error": "Failed to generate image"}), 500


def initialize_current_sd_model_state():
    """Fetches the currently loaded model from SD API on startup."""
    global MODEL
    
    try:
        logging.info("Attempting to fetch current SD model state from API...")
        response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=10)
        response.raise_for_status()
        options = response.json()
        current_sd_model = options.get("sd_model_checkpoint", "Unknown")
        MODEL = current_sd_model
        logging.info(f"Successfully initialized current SD model to: {MODEL}")
                                                 
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch current SD model state from API on startup: {e}. MODEL will remain 'Unknown'.")
        MODEL = "Unknown"
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    initialize_current_sd_model_state()
                                  

    api_worker_thread = threading.Thread(target=api_worker)
    api_worker_thread.daemon = True
    api_worker_thread.start()
    logging.info("Started Stable Diffusion API worker thread.")

    start_cleanup_thread()
    app.run(host='0.0.0.0', debug=True)