"""
video_to_audio_converter.py
---------------------------
Converts all .mp4 videos in 'raw_data/' into .wav or .mp3 audio files.
Outputs are saved in the SAME folder as the videos.

✅ Compatible with both MoviePy 1.x and 2.x
✅ Runs directly in PyCharm (no arguments needed)
✅ Skips files already converted

Usage:
    Just run inside PyCharm → Run ▶ video_to_audio_converter
"""

from pathlib import Path
from tqdm import tqdm

# --- Import VideoFileClip (handles both MoviePy 1.x and 2.x) ---
try:
    from moviepy.editor import VideoFileClip   # For MoviePy 1.x
except ModuleNotFoundError:
    from moviepy import VideoFileClip          # For MoviePy 2.x


def convert_videos_to_audio(folder: str = "raw_data", audio_format: str = "wav"):
    """
    Converts all .mp4 videos in 'folder' to audio files (.wav or .mp3)
    and saves them in the same folder.
    """
    # --- Setup paths ---
    root_dir = (
        Path(__file__).resolve().parent.parent
        if Path(__file__).resolve().parent.name == "scripts"
        else Path(__file__).resolve().parent
    )
    data_path = root_dir / folder

    if not data_path.exists():
        print(f"❌ Folder not found: {data_path}")
        return

    # --- Find video files ---
    video_files = list(data_path.glob("*.mp4"))
    if not video_files:
        print(f"⚠️ No .mp4 files found in '{data_path}'.")
        return

    print(f"\n🎬 Found {len(video_files)} video(s) in '{data_path}'.")
    print(f"🎧 Extracting audio as {audio_format.upper()}...\n")

    # --- Conversion loop ---
    for video_file in tqdm(video_files, desc="Processing videos", unit="file"):
        audio_filename = video_file.stem + f".{audio_format}"
        audio_path = data_path / audio_filename

        # Skip if already converted
        if audio_path.exists():
            print(f"⏭️  Skipping {audio_filename} (already exists)")
            continue

        try:
            clip = VideoFileClip(str(video_file))

            # MoviePy 1.x uses verbose/logger args; 2.x does not
            try:
                clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            except TypeError:
                clip.audio.write_audiofile(str(audio_path))

            clip.close()
        except Exception as e:
            print(f"⚠️ Error processing {video_file.name}: {e}")

    print(f"\n✅ Conversion complete! Audio files saved in: {data_path}\n")


if __name__ == "__main__":
    convert_videos_to_audio(audio_format="wav")  # Change to "mp3" if needed
