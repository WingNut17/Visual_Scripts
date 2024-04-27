import subprocess
import numpy as np
import cv2
import os
import json
import random
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from scipy import signal

def run_ffmpeg_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg command: {e}")
        return False
    return True

def get_video_resolution(file_path):
    try:
        ffprobe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', file_path]
        result = subprocess.run(ffprobe_command, capture_output=True, text=True)
        metadata = json.loads(result.stdout)

        for stream in metadata['streams']:
            if stream.get('codec_type') == 'video':
                width = stream.get('width')
                height = stream.get('height')
                return width, height

        print("No video stream found in the file.")
        return None, None

    except Exception as e:
        print(f"Error extracting resolution: {e}")
        return None, None

def get_video_framerate(file_path):
    try:
        ffprobe_command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', file_path]
        result = subprocess.run(ffprobe_command, capture_output=True, text=True)
        metadata = json.loads(result.stdout)

        for stream in metadata['streams']:
            if stream.get('codec_type') == 'video':
                framerate = eval(stream.get('r_frame_rate'))
                return framerate

        print("No video stream found in the file.")
        return None

    except Exception as e:
        print(f"Error extracting framerate: {e}")
        return None

def calculate_num_frames(yuv_file, width, height):
    frame_size = int(1.5 * width * height)

    file_size = os.path.getsize(yuv_file)

    num_frames = file_size // frame_size

    return num_frames

def extract_frame_info(yuv_data, width, height):
    frame_size = width * height * 3 // 2

    Y = yuv_data[:frame_size]
    U = yuv_data[frame_size:frame_size + (width * height // 4)]
    V = yuv_data[frame_size + (width * height // 4):]

    return Y, U, V

def line_glitch(frame_data, width, height):
    Y, U, V = separate_YUV_into_Y_U_V(height, width, frame_data)

    for i in range(height):
        glitch_row = random.randint(0, height - 1)
        if glitch_row == i:
            saturation_factor = random.uniform(1.0, 1.1)  # Randomly choose a saturation factor
            Y[i * width : (i + 1) * width] = np.clip(Y[i * width : (i + 1) * width] ** saturation_factor, 0, 255)
            U[i * width : (i + 1) * width] = np.clip(U[i * width : (i + 1) * width] * saturation_factor, 0, 255)
            V[i * width : (i + 1) * width] = np.clip(V[i * width : (i + 1) * width] * saturation_factor, 0, 255)

    return combine_Y_U_V_into_YUV(Y, U, V)

def combine_Y_U_V_into_YUV(Y, U, V):
    frame_data = bytearray()
    frame_data.extend(Y.tobytes())
    frame_data.extend(U.tobytes())
    frame_data.extend(V.tobytes())
    return frame_data

def generate_random_noise(height, width, num_frames, yuv_data, strength):
    for i in range(num_frames):
        start_index = i * height * width * 3 // 2
        end_index = (i + 1) * height * width * 3 // 2
        yuv_frame = yuv_data[start_index:end_index]

        noise = np.random.randint(-strength, strength, size=yuv_frame.shape)
        yuv_frame = np.clip(yuv_frame + noise, 0, 255)

        yuv_data[start_index:end_index] = yuv_frame

    return yuv_data

def generate_random_shift(height, width, num_frames, yuv_data, strength):
    for i in range(num_frames):
        start_index = i * height * width * 3 // 2
        end_index = (i + 1) * height * width * 3 // 2
        yuv_frame = yuv_data[start_index:end_index]

        if len(yuv_frame) != height * width * 3 // 2:
            print(f"Frame {i} has unexpected size: {len(yuv_frame)}, expected: {height * width * 3 // 2}")
            continue

        yuv_frame = yuv_frame.reshape((height * 3 // 2, width))

        for row in range(height):
            shift_direction = np.random.choice([-strength, strength])
            
            yuv_frame[row][:width-1] = np.roll(yuv_frame[row][:width-1], shift_direction)

        yuv_data[start_index:end_index] = yuv_frame.flatten()

    return yuv_data

def apply_sinusoidal_shift(frame_data, height, width, frame_index, num_frames, strength):
    frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((height * 3 // 2, width)).copy()

    phase_shift = strength * np.pi * frame_index / num_frames

    for row in range(height):
        offset = int(np.sin(row / height * 100 * np.pi + phase_shift) * strength * width / 100) 
        
        frame_array[row] = np.roll(frame_array[row], offset)

    return frame_array.tobytes()

def wobble_video_horizontally(frame_data, height, width, num_frames, strength):
    frame_size = height * width * 3 // 2 

    yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

    Y = yuv_frame[:height * width].reshape((height, width))
    U = yuv_frame[height * width : frame_size * 5 // 6].reshape((height // 2, width // 2))
    V = yuv_frame[frame_size * 5 // 6 :].reshape((height // 2, width // 2))

    Y_modified = apply_sinusoidal_shift(Y.tobytes(), height, width, num_frames, num_frames, strength)

    yuv_frame_modified = np.concatenate((np.frombuffer(Y_modified, dtype=np.uint8), U.flatten(), V.flatten()))

    return yuv_frame_modified.tobytes()

def alter_saturation(frame_size, yuv_data, saturation_factor):
    yuv_array = np.frombuffer(yuv_data, dtype=np.uint8).copy()

    Y = np.array(yuv_array[:frame_size * 2 // 3])
    U = np.array(yuv_array[frame_size * 2 // 3 : frame_size * 5 // 6])
    V = np.array(yuv_array[frame_size * 5 // 6 :])

    U_mean = np.mean(U)
    V_mean = np.mean(V)
    U_delta = U - U_mean
    V_delta = V - V_mean
    U_modified = U_mean + saturation_factor * U_delta
    V_modified = V_mean + saturation_factor * V_delta

    U_modified = np.clip(U_modified, 0, 255)
    V_modified = np.clip(V_modified, 0, 255)

    yuv_array[frame_size * 2 // 3 : frame_size * 5 // 6] = U_modified
    yuv_array[frame_size * 5 // 6 :] = V_modified

    return yuv_array.tobytes()

def get_info(input_file_path, output_file_path):
    width, height = get_video_resolution(input_file_path)
    framerate = get_video_framerate(input_file_path)
    num_frames = calculate_num_frames(output_file_path,width,height)
    return width, height, framerate, num_frames

def separate_YUV_into_Y_U_V(height, width, frame_data):
    Y_size = height * width
    Y = np.frombuffer(frame_data[:Y_size], dtype=np.uint8).copy()  # Make a copy to ensure writeability
    U = np.frombuffer(frame_data[Y_size : Y_size + Y_size // 4], dtype=np.uint8).copy()
    V = np.frombuffer(frame_data[Y_size + Y_size // 4 :], dtype=np.uint8).copy()
    return Y, U, V

def read_raw_audio(file_path, height, width, num_frames):
    dtype = np.uint8
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    if len(raw_data) == 0:
        raise ValueError("Raw audio data is empty")
    
    audio_data = np.frombuffer(raw_data, dtype=dtype)
    
    audio_data = audio_data.reshape((-1, 1))

    audio_data = audio_data - np.min(audio_data)

    yuv_file_len = (height * width * 3 // 2) * num_frames

    if len(audio_data) < yuv_file_len:
        audio_data = np.tile(audio_data, (yuv_file_len // len(audio_data), 1))
        audio_data = np.vstack([audio_data, audio_data[:yuv_file_len % len(audio_data)]])
    elif len(audio_data) > yuv_file_len:
        audio_data = audio_data[:yuv_file_len]

    return audio_data

def combine_yuv_audio(frame_data, audio_data, current_frame):
    combined_data = []
    for i in range(len(frame_data)):
        combined_frame = np.clip((frame_data[i] + (audio_data[current_frame*len(frame_data) + i])), 0, 255)
        combined_data.append(combined_frame)
    return np.array(combined_data)

def jitter_frame_up(height, yuv_frame, strength):
    try:
        if yuv_frame is None:
            raise ValueError("Empty frame data received.")

        frame_array = np.frombuffer(yuv_frame, dtype=np.uint8)

        frame_array = frame_array.reshape((height * 3 // 2, -1))

        jitter_amount = int(np.random.exponential(scale=strength))

        jitter_amount = min(jitter_amount, height - 1)

        jittered_frame_array = np.roll(frame_array, -jitter_amount, axis=0)

        jittered_frame_data = jittered_frame_array.tobytes()

        return jittered_frame_data
    except Exception as e:
        print(f"Error in jitter_frame_up: {e}")
        return None 


directory = 'E:\\Art\\Editing\\EDITS\\Mar24\\data_bending\\'
input_file = 'input1.mp4'
input_file2 = 'input2.mp4'
output_file = 'output1.yuv'
audio_file = 'E:\\Art\\Editing\\EDITS\\Mar24\\data_bending\\audio.raw'

for file in [output_file, f"edited_{output_file}", "edited_output.mp4"]:
    try:
        os.remove(os.path.join(directory, file))
        print(f"Successfully deleted {file}.")
    except FileNotFoundError:
        print(f"{file} not found.")


ffmpeg_command = f"ffmpeg -i {directory}{input_file} -pix_fmt yuv420p {directory}{output_file}"
if run_ffmpeg_command(ffmpeg_command):
    print("Conversion to yuv successful!")
else:
    print("Conversion failed.")

# Read a YUV file
width, height, framerate, num_frames = get_info(f"{directory}{input_file}", f"{directory}{output_file}")
frame_size = width * height * 3 // 2

# Read yuv_data from the file
with open(os.path.join(directory, f"{directory}{output_file}"), 'rb') as f:
    yuv_data = f.read()

# Process each frame and write to the output file
with open(os.path.join(directory, f"{directory}all_frames.yuv"), 'wb') as all_frames_file:
    for i in range(num_frames):
        start_index = i * frame_size
        end_index = start_index + frame_size

        # Extract frame data
        frame_data = yuv_data[start_index:end_index]
        frame_data = np.frombuffer(frame_data, dtype=np.uint8)

        # Apply sinusoidal shift to the entire frame data
        #frame_data_modified = modify(height, width, frame_data)
        #frame_data_modified = line_glitch(frame_data, width, height)
        #frame_data_modified = glitch_warp(width, height, frame_data)
        frame_data_modified = jitter_frame_up(height,frame_data,1)
        frame_data_modified = alter_saturation(frame_size,frame_data_modified,5)
        frame_data_modified = wobble_video_horizontally(frame_data_modified,height,width,num_frames,4)

        # Write modified frame data directly to the output file
        all_frames_file.write(frame_data_modified)

#yuv to mp4
ffmpeg_command = f"ffmpeg -s {width}x{height} -pix_fmt yuv420p -i {directory}all_frames.yuv -c:v libx264 -preset medium -crf 1 -r {framerate} {directory}edited_output.mp4"
if run_ffmpeg_command(ffmpeg_command):
    print("Conversion back to mp4 successful!")
else:
    print("Conversion failed.")