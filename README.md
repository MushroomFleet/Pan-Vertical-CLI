# 🎬✨ VIDEO PANNER 3000 ✨🎬

> _Transform your boring landscape videos into **EPIC** portrait masterpieces!_ 🔄🔥🔥🔥

## 🤔 What is this madness?! 🤯

VIDEO PANNER 3000 is a **REVOLUTIONARY** 🚀 tool that transforms your 16:9 landscape videos (1280x720) into 9:16 portrait format (720x1280) with SMOOTH 🧈 panning effects - perfect for social media content! 📱✨

Instead of cropping out important parts of your video or having ugly black bars 🖤, VIDEO PANNER 3000 intelligently moves the frame across your video with customizable animation points! 🎯🎯🎯

### 🌟 INCREDIBLE FEATURES 🌟

- 🎭 Convert ANY landscape video to portrait format with intelligent panning
- 🎮 Define EXACTLY where your camera movements happen with custom markpoints
- 🔄 Batch process ENTIRE FOLDERS of videos with ONE command 😱
- 🔊 Preserves original audio from your source videos - no silent movies! 🎵
- 🏷️ Add your own watermarks or branding with custom overlays
- ⏱️ Automatic timestamp appending to output filenames
- 📊 Progress tracking with ETA (because waiting is BORING 😴)

## 🛠️ Installation 🛠️

### Step 1: Clone this AMAZING repository 🤩

```bash
git clone https://github.com/MushroomFleet/Pan-Vertical-CLI
cd video-panner-3000
```

### Step 2: Install dependencies 📦📦📦

```bash
# Create a virtual environment (HIGHLY recommended! 👍)
python -m venv venv

# Activate the virtual environment
# For Windows 🪟
venv\Scripts\activate
# For macOS/Linux 🐧
source venv/bin/activate

# Install the dependencies 🔽
pip install -r requirements.txt
# or install manually:
pip install opencv-python numpy ffmpeg-python

# Install FFmpeg (system dependency) 🎬
# For Windows:
# Download from https://ffmpeg.org/download.html or use Chocolatey:
choco install ffmpeg
# For macOS:
brew install ffmpeg
# For Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg
```

## 🚀 How to Use It 🚀

VIDEO PANNER 3000 is SUPER EASY to use! 🙌 Just follow these simple steps:

### 1️⃣ Create a Configuration File 📝

Create a `config.json` file with your desired settings:

```json
{
    "source": "/path/to/videos/",
    "output": "/path/to/output/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "none"
}
```

### 2️⃣ Run the Script 🏃‍♂️💨

```bash
python video_panner.py --config config.json
```

### 3️⃣ ENJOY YOUR AMAZING VIDEOS! 🎉🥳🎊

## 📋 Configuration Options Explained 📋

Your `config.json` file is the COMMAND CENTER for your video transformations! 🎛️

| Setting | Description | Example |
|---------|-------------|---------|
| `source` 📂 | Path to input video file OR folder for batch processing | `"/videos/my_cool_video.mp4"` or `"/videos/"` |
| `output` 💾 | Where to save your TRANSFORMED videos | `"/processed_videos/"` |
| `markpoint1` ⬅️ | When to START panning (% of video duration) | `5.0` (starts at 5% of video) |
| `markpoint2` ⏺️ | When to reach CENTER position (% of video duration) | `50.0` (centered at half-way point) |
| `markpoint3` ➡️ | When to reach RIGHT position (% of video duration) | `90.0` (right-aligned at 90% of video) |
| `overlay` 🖼️ | Overlay image filename (place in `/templates/` folder) or "none" | `"mylogo.png"` or `"none"` |

## 🎯 Step-by-Step Guide to PERFECTION 🎯

### 🔍 Processing a Single Video 🔍

1. Create your `config.json`:
```json
{
    "source": "/videos/awesome_landscape_video.mp4",
    "output": "/videos/processed/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "none"
}
```

2. Run the command:
```bash
python video_panner.py --config config.json
```

3. 🎉 Your video will be saved as `/videos/processed/awesome_landscape_video_YYYYMMDD_HHMMSS.mp4`!

### 📁 Batch Processing an ENTIRE FOLDER 📁

1. Create your `config.json` pointing to a FOLDER:
```json
{
    "source": "/videos/vacation_videos/",
    "output": "/videos/processed/",
    "markpoint1": 10.0,
    "markpoint2": 40.0,
    "markpoint3": 85.0,
    "overlay": "none"
}
```

2. Run the command:
```bash
python video_panner.py --config config.json
```

3. 🎉 ALL your videos will be processed with the SAME panning effect!

### 🖼️ Adding Your AWESOME Brand Overlay 🖼️

1. Create a transparent PNG image with your branding
2. Save it in the `/templates/` folder (will be created automatically)
3. Update your `config.json`:
```json
{
    "source": "/videos/product_demo.mp4",
    "output": "/videos/branded/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "my_brand_logo.png"
}
```

4. Run the command:
```bash
python video_panner.py --config config.json
```

5. 🎉 Your video now has your branding overlaid throughout!

## 🛠️ Command Line Options 🛠️

### `--config` (REQUIRED) 📄

Tell the script where to find your configuration file.

```bash
python video_panner.py --config my_special_config.json
```

### `--verbose` (OPTIONAL) 🔊

Shows DETAILED progress information - for when you're SUPER impatient! 😜

```bash
python video_panner.py --config config.json --verbose
```

## 🔮 Advanced Examples 🔮

### 🔥 Quick Panning Effect 🔥

```json
{
    "source": "/videos/action_scene.mp4",
    "output": "/videos/processed/",
    "markpoint1": 10.0,
    "markpoint2": 30.0,
    "markpoint3": 60.0,
    "overlay": "none"
}
```
This configuration creates a FASTER panning effect by making the camera reach the right side earlier (60% instead of 90%)! ⚡️

### 🐢 Slow, Cinematic Panning 🐢

```json
{
    "source": "/videos/nature_timelapse.mp4",
    "output": "/videos/cinematic/",
    "markpoint1": 20.0,
    "markpoint2": 60.0,
    "markpoint3": 95.0,
    "overlay": "film_grain.png"
}
```
This creates a SLOWER, more DRAMATIC panning effect with a film grain overlay! 🎭

### 💼 Batch Processing with Corporate Branding 💼

```json
{
    "source": "/videos/product_demos/",
    "output": "/videos/social_media_ready/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "corporate_logo_transparent.png"
}
```
Process ALL your product demos with your corporate branding! 📈

## ⚠️ Limitations ⚠️

- 📱 ONLY outputs 720x1280 portrait videos
- ⏱️ Recommended for videos under 2 minutes (but will work with longer videos if you're PATIENT! ⏳)
- 🚫 No zooming functionality (to keep things SIMPLE! 👌)

## 🐛 Troubleshooting 🐛

### 🔴 ERROR: "Could not open video file" 🔴

- Check that your video path is correct 🔍
- Make sure the video file exists and isn't corrupted 🤕
- Verify your video is in a supported format (mp4, mov, avi, mkv) 📼

### 🔵 ERROR: "Could not create output video writer" 🔵

- Check that your output directory exists or can be created 📁
- Make sure you have write permissions to that location 🔒
- Verify you have enough disk space 💾

## 🤝 Contributing 🤝

CONTRIBUTIONS are WELCOME! 🎁 Feel free to submit a pull request or open an issue for:

- 🐞 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 UI enhancements

## 📜 License 📜

This project is licensed under the MIT License - see the LICENSE file for details. 📄

## 🙏 Acknowledgments 🙏

- 🧠 OpenCV for the amazing video processing capabilities
- 💻 NumPy for handling all the math stuff
- 🎬 FFmpeg for powerful audio/video handling capabilities
- 🌎 The open-source community for their ENDLESS inspiration

---

Made with ❤️ (and WAY TOO MANY EMOJIS 🤪) by DRIFT JOHNSON
