import os
import sys
import random
import librosa
import soundfile as sf

def preprocess_wav_files(base_dir):
    background_noise_dir = os.path.join(base_dir, "_background_noise_")
    silence_dir = os.path.join(base_dir, "_silence_")
    validation_file = os.path.join(base_dir, "validation_list.txt")
    testing_file = os.path.join(base_dir, "testing_list.txt")

    os.makedirs(silence_dir, exist_ok=True)

    clip_paths = []
    for filename in os.listdir(background_noise_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(background_noise_dir, filename)
            
            audio, sr = librosa.load(file_path, sr=None)
            duration_seconds = len(audio) // sr

            for i in range(duration_seconds):
                start_sample = i * sr
                end_sample = (i + 1) * sr
                clip = audio[start_sample:end_sample]
                
                clip_filename = f"{os.path.splitext(filename)[0]}_{i}.wav"
                clip_path = os.path.join(silence_dir, clip_filename)
                clip_paths.append(os.path.join("_silence_", clip_filename))
                sf.write(clip_path, clip, sr)

    num_clips = len(clip_paths)
    sample_size = min(260, num_clips // 10)

    validation_sample = random.sample(clip_paths, sample_size)
    remaining_clips = list(set(clip_paths) - set(validation_sample))
    testing_sample = random.sample(remaining_clips, sample_size)

    # while len(validation_sample) < 260:
    #     validation_sample.append(random.choice(validation_sample))
    # while len(testing_sample) < 260:
    #     testing_sample.append(random.choice(testing_sample))

    with open(validation_file, "a") as vf:
        vf.write("\n".join(validation_sample))
    with open(testing_file, "a") as tf:
        tf.write("\n".join(testing_sample))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_dataset.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    preprocess_wav_files(base_directory)