import cv2
import numpy as np
import random
import os
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d, convolve

def main():
    """Currently not up to date. Just the basis for future ui."""
    

    choice = ""
    effects = []
    video_path = ""

    while not os.path.exists(video_path):
        video_path = input(
            "Welcome to the video glitch program.\n"
            "Please enter the file path for the video you'd like to glitch: ('E:\\Art\\Editing\\data_bending\\input1.mp4')\n"
        )

        # check that video file exists
        if not os.path.exists(video_path):
            print("File does not exist. Please enter a valid file path.")

    # add the video file location
    effects.append(video_path)

    while choice != "q":
        choice = input(
            "What effect would you like to apply:\n"
            "1) Line glitch\n"
            "2) Noise\n"
            "3) Horizontal Wave\n"
            "4) Saturation\n"
            "5) Audio Glitch\n"
            "('f' to finish adding glitch effects, or 'q' to quit.)\n"
        )

        if choice == "f":
            if len(effects) < 2:
                print("You need to select at least one effect.")
            else:
                print("Effects to be applied:", effects)
                break
        elif choice == "q":
            quit()

        try:
            effect_choice = int(choice)
            if effect_choice == 1:
                strength_saturation = input(
                    "Enter the strength and saturation factor for the lines between 1-100: (ex: '36,52')\n"
                )
                strength, saturation = map(int, strength_saturation.strip().split(','))
                effects.append((effect_choice, np.clip(strength/100,0,100), np.clip(saturation,0,100)))

            elif effect_choice == 2:
                strength = int(input("Enter the strength factor: (ex: '75')\n"))
                effects.append((effect_choice, strength))

            elif effect_choice == 3:
                strength_saturation = input(
                    "Enter the strength and speed factor for the wave between 1-100: (ex: '82,9')\n"
                )
                strength, speed_factor = map(int, strength_saturation.strip().split(','))
                effects.append((effect_choice, np.clip(strength,0,100), np.clip(speed_factor,0,100)))

            elif effect_choice == 4:
                saturation = int(input("Enter the saturation factor: (ex: '75')\n"))
                effects.append((effect_choice, np.clip(saturation,0,100)))

            elif effect_choice == 5:
                while not os.path.exists(audio_file):
                    audio_file = input(
                    "Enter the audio file path: (ex: 'E:\\Art\\Editing\\data_bending\\input_audio.mp3')\n"
                    )

                    if not os.path.exists(audio_file):
                        print("Audio file does not exist. Please enter a valid file path.")
                    else:
                        effects.append((effect_choice, audio_file))
                        break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")

        except ValueError:
            if choice != "q" and choice != "f":
                print("Invalid input. Please enter a valid number, 'f', or 'q' to quit.")

        except Exception as e:
            print(f"Error: {e}")

    process_video(effects)


def get_video_info(file_path):
    """Gets and returns the basic video information, given the file path."""

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {file_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    _, frame = cap.read()
    channels = frame.shape[2] if frame is not None else 3

    cap.release()

    return width, height, framerate, num_frames, channels


def line_glitch(frame, saturation_factor, height, strength):
    """Applies a line glitch to the frame, choosing a random line.

    frame: frame data in BRG format with shape (height, width, 3)
    saturation factor: how much to saturate the line
    strength: 0-1 float. Represents roughly the % chance of glitching each line"""

    for i in range(height):
        if random.random() < strength:
            x = random.randint(0,2)
            if x == 0:
                frame[i, :, 0] = np.clip(frame[i, :, 0] * saturation_factor, 0, 255)    # saturate the B pixel data
            elif x == 1:
                frame[i, :, 1] = np.clip(frame[i, :, 1] * saturation_factor, 0, 255)    # saturate the R pixel data
            elif x == 2:
                frame[i, :, 2] = np.clip(frame[i, :, 2] * saturation_factor, 0, 255)    # saturate the G pixel data
    return frame


def generate_random_noise(frame, strength):
    """Simply applies random noise to the given frame with a given strength"""

    noise = np.random.randint(-strength, strength, frame.shape, dtype='int16')
    frame = frame.astype('int16') + noise
    return np.clip(frame, 0, 255).astype('uint8')


def apply_sinusoidal_shift(frame, frame_index, num_frames, strength, speed_factor):
    """Shifts the rows"""
    
    height, width, _ = frame.shape
    phase_shift = strength * np.pi * frame_index * speed_factor / num_frames

    for row in range(height):
        offset = int(np.sin(row / height * 100 * np.pi + phase_shift) * strength * width / 100)
        frame[row] = np.roll(frame[row], offset, axis=0)

    return frame


def blur(frame):
    frame = frame.flatten()

    altered_frame = gaussian_filter1d(frame, sigma=4)

    frame = altered_frame.reshape(180,320,3)

    return frame


def reverb(frame, kernel):

    frame = frame.flatten()

    altered_frame = convolve(frame, kernel, mode='reflect')

    frame = altered_frame.reshape(180,320,3)

    return frame


def hue_glitch(frame, height):
    """Applies a line glitch to the frame, choosing a random line.

    frame: frame data in BRG format with shape (height, width, 3)
    saturation factor: how much to saturate the line
    strength: 0-1 float. Represents roughly the % chance of glitching each line"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hue = 1

    for i in range(height):
        hsv[i, :, 0] = np.clip(hsv[i, :, 0] + 2 * hue, 0, 255)
        hsv[i, :, 2] = np.clip(hsv[i, :, 2] * 1.5, 0, 255)

        hue += 1
        
        if hue == 255:
            hue = 0
    
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame


def alter_saturation(frame, saturation_factor):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return frame


def read_mp3_audio(audio_input, height, width, num_frames):
    audio = AudioSegment.from_file(audio_input)
    audio_data = np.array(audio.get_array_of_samples())

    target_size = height * width * 3 * num_frames

    if len(audio_data) < target_size:
        # Duplicate the audio data to match the target size
        repetitions = target_size // len(audio_data) + 1
        audio_data = np.tile(audio_data, repetitions)
        audio_data = audio_data[:target_size]
    elif len(audio_data) > target_size:
        # Trim the audio data to match the target size
        audio_data = audio_data[:target_size]

    return audio_data


def combine_audio_with_video_frame(frame, audio_data, frame_index, height, width):
    frame_flat = frame.reshape(-1, 3)  # Flatten the frame to a 2D array (pixels x 3 channels)
    audio_slice = audio_data[frame_index * height * width * 3 : (frame_index + 1) * height * width * 3]  # Extract audio for this frame

    audio_reshaped = audio_slice.reshape(-1, 3)  # Reshape audio to match the flattened frame

    combined_frame = np.clip(frame_flat.astype(np.int16) + (audio_reshaped.astype(np.int16))//200, 0, 255).astype(np.uint8)  # Combine audio and video

    edited_frame = combined_frame.reshape(height, width, 3)  # Reshape back to original shape

    return edited_frame


def process_video(effects):
    input_file = effects[0]
    
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file {input_file}")

    width, height, framerate, num_frames, channels = get_video_info(input_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    input_directory = os.path.dirname(input_file)
    output_path = os.path.join(input_directory, "output_video.mp4")

    out = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))

    audio_data = None

    if any(effect[0] == 5 for effect in effects):
        audio_file = next(effect[1] for effect in effects if effect[0] == 5)
        audio_data = read_mp3_audio(audio_file, height, width, num_frames)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for effect in effects[1:]:
            effect_choice = effect[0]
            if effect_choice == 1:
                strength, saturation = effect[1], effect[2]
                frame = line_glitch(frame, saturation, height, strength)
            elif effect_choice == 2:
                strength = effect[1]
                frame = generate_random_noise(frame, strength)
            elif effect_choice == 3:
                strength, speed_factor = effect[1], effect[2]
                frame = apply_sinusoidal_shift(frame, frame_index, num_frames, strength, speed_factor)
            elif effect_choice == 4:
                saturation = effect[1]
                frame = alter_saturation(frame, saturation)
            elif effect_choice == 5:
                frame = combine_audio_with_video_frame(frame, audio_data, frame_index, height, width)

        out.write(frame)
        frame_index += 1
        print(f"Edited frame {frame_index} of {num_frames}, {round(100*(frame_index/num_frames), 4)}%")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #main()
    
    input_file = "E:\\Art\\Editing\\EDITS\\Mar24\\data_bending\\input1.mp4"
    
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file {input_file}")

    width, height, framerate, num_frames, channels = get_video_info(input_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    input_directory = os.path.dirname(input_file)
    output_path = os.path.join(input_directory, "output_video.mp4")

    out = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))

    #audio_file = "E:\\Art\\Editing\\EDITS\\Mar24\\data_bending\\input_audio.mp3"
    #audio_data = read_mp3_audio(audio_file, height, width, num_frames)

    # used for testing convolve
    kernel_size = 11
    sigma = 10.0
    x = np.linspace(-sigma, sigma, kernel_size)
    gauss_kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel_1d /= gauss_kernel_1d.sum()

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #frame = line_glitch(frame, 2, height, 0.01)

        #frame = apply_sinusoidal_shift(frame, frame_index, num_frames, 2, 2)

        #frame = blur(frame)

        frame = reverb(frame, gauss_kernel_1d)

        #frame = alter_saturation(frame, 3)

        out.write(frame)
        frame_index += 1

        print(f"Edited frame {frame_index} of {num_frames-1}, {round(100*(frame_index/(num_frames-1)), 1)}%")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
