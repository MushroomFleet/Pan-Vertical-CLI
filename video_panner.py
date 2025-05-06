#!/usr/bin/env python3
"""
Video Panning Tool - Convert landscape videos to portrait with panning effects.

This script transforms 1280x720 landscape videos into 720x1280 portrait videos
with a configurable panning effect that moves from left to right through
user-defined markpoints.

Usage:
    python video_panner.py --config path/to/config.json
"""

import os
import sys
import json
import argparse
import time
import subprocess
from datetime import datetime
import cv2
import numpy as np
import ffmpeg

# Constants
DEFAULT_MARKPOINT1 = 5.0   # 5% into video duration, left align
DEFAULT_MARKPOINT2 = 50.0  # 50% into video duration, center align
DEFAULT_MARKPOINT3 = 90.0  # 90% into video duration, right align
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280

def load_config(config_path):
    """Load and validate the configuration file."""
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # Validate required fields
        required_fields = ['source', 'output']
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field '{field}' in config file")
                sys.exit(1)
        
        # Set default values for optional fields
        if 'markpoint1' not in config:
            config['markpoint1'] = DEFAULT_MARKPOINT1
        if 'markpoint2' not in config:
            config['markpoint2'] = DEFAULT_MARKPOINT2
        if 'markpoint3' not in config:
            config['markpoint3'] = DEFAULT_MARKPOINT3
        if 'overlay' not in config:
            config['overlay'] = 'none'
        
        # Validate markpoint values
        try:
            mp1 = float(config['markpoint1'])
            mp2 = float(config['markpoint2'])
            mp3 = float(config['markpoint3'])
            
            if not (0 <= mp1 < mp2 < mp3 <= 100):
                print("Error: Markpoints must be in ascending order and between 0 and 100")
                sys.exit(1)
                
        except ValueError:
            print("Error: Markpoints must be numeric values")
            sys.exit(1)
            
        return config
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in config file")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)

def ensure_templates_directory():
    """Create templates directory if it doesn't exist."""
    if not os.path.exists('templates'):
        os.makedirs('templates')
        with open(os.path.join('templates', 'templates-here.txt'), 'w') as f:
            f.write("Place your template overlay files in this directory.")
        print("Created 'templates' directory")

def process_video(config, verbose=False):
    """Process a single video to apply panning effect."""
    source_path = config['source']
    output_path = config['output']
    markpoint1 = float(config['markpoint1']) / 100.0  # Convert percentage to decimal
    markpoint2 = float(config['markpoint2']) / 100.0
    markpoint3 = float(config['markpoint3']) / 100.0
    
    try:
        # Open the source video
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {source_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if verbose:
            print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Ensure the input video is the expected dimensions
        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            print(f"Warning: Input video dimensions ({width}x{height}) differ from expected ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
        
        # Calculate frame indices for markpoints
        markpoint1_frame = int(total_frames * markpoint1)
        markpoint2_frame = int(total_frames * markpoint2)
        markpoint3_frame = int(total_frames * markpoint3)
        
        if verbose:
            print(f"Markpoint frames: {markpoint1_frame}, {markpoint2_frame}, {markpoint3_frame}")
        
        # Create output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, f"{name}_{timestamp}.mp4")
        
        # Set up the output video writer - 720x1280 (portrait)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        
        if not out.isOpened():
            print(f"Error: Could not create output video writer")
            cap.release()
            return None
        
        # Load overlay if specified
        overlay = None
        if config['overlay'] != 'none':
            template_path = os.path.join('templates', config['overlay'])
            if os.path.exists(template_path):
                overlay = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    print(f"Warning: Could not load overlay image at {template_path}")
                elif verbose:
                    print(f"Loaded overlay: {template_path}, shape: {overlay.shape}")
            else:
                print(f"Warning: Overlay image not found at {template_path}")
        
        # Process each frame
        current_frame = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate the x-offset for panning effect
            if current_frame < markpoint1_frame:
                # Before markpoint1: left-aligned
                x_offset = 0
            elif current_frame < markpoint2_frame:
                # Between markpoint1 and markpoint2: animate from left to center
                progress = (current_frame - markpoint1_frame) / max(1, (markpoint2_frame - markpoint1_frame))
                max_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                x_offset = int(progress * max_offset)
            elif current_frame < markpoint3_frame:
                # Between markpoint2 and markpoint3: animate from center to right
                progress = (current_frame - markpoint2_frame) / max(1, (markpoint3_frame - markpoint2_frame))
                start_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                end_offset = width - OUTPUT_WIDTH  # Right-aligned position
                x_offset = int(start_offset + (progress * (end_offset - start_offset)))
            else:
                # After markpoint3: right-aligned
                x_offset = width - OUTPUT_WIDTH
            
            # Ensure x_offset is within bounds
            x_offset = max(0, min(width - OUTPUT_WIDTH, x_offset))
            
            # Crop the frame to get the 720 wide section
            cropped_frame = frame[:, int(x_offset):int(x_offset+OUTPUT_WIDTH), :]
            
            # Create portrait frame (720x1280)
            portrait_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            
            # Place cropped frame in the middle of the portrait frame vertically
            y_offset = (OUTPUT_HEIGHT - height) // 2
            portrait_frame[y_offset:y_offset+height, :, :] = cropped_frame
            
            # Add overlay if available
            if overlay is not None:
                # Resize overlay to match output dimensions
                overlay_resized = cv2.resize(overlay, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                
                # Apply alpha blending for transparent overlay
                if overlay_resized.shape[2] == 4:  # With alpha channel
                    # Split the overlay into BGR and alpha channels
                    bgr = overlay_resized[:, :, 0:3]
                    alpha = overlay_resized[:, :, 3]
                    
                    # Create a mask and normalize
                    alpha_norm = alpha / 255.0
                    
                    # Apply alpha blending for each channel
                    for c in range(3):
                        portrait_frame[:, :, c] = ((1.0 - alpha_norm) * portrait_frame[:, :, c] + 
                                                 alpha_norm * bgr[:, :, c]).astype(np.uint8)
                else:  # Without alpha channel
                    portrait_frame = cv2.addWeighted(portrait_frame, 1, overlay_resized, 0.3, 0)
            
            # Write the frame to output video
            out.write(portrait_frame)
            
            current_frame += 1
            
            # Show progress
            if current_frame % 100 == 0 or current_frame == total_frames:
                progress_percent = int(current_frame/total_frames*100)
                elapsed_time = time.time() - start_time
                frames_per_second = current_frame / max(1, elapsed_time)
                remaining_frames = total_frames - current_frame
                estimated_remaining_time = remaining_frames / max(1, frames_per_second)
                
                if verbose:
                    print(f"Processing {filename}: {current_frame}/{total_frames} frames ({progress_percent}%) - "
                          f"{frames_per_second:.2f} fps, ETA: {estimated_remaining_time:.2f}s")
                else:
                    print(f"Processing {filename}: {progress_percent}% complete, ETA: {estimated_remaining_time:.2f}s")
        
        # Release resources
        cap.release()
        out.release()
        
        elapsed_time = time.time() - start_time
        print(f"Processing complete for {filename}. Silent video saved to {output_file} (took {elapsed_time:.2f}s)")
        
        # Create a path for the final video with audio
        name_parts = os.path.splitext(output_file)
        final_output = f"{name_parts[0]}_with_audio{name_parts[1]}"
        
        # Add audio from source video to the processed video
        print(f"Adding audio from source to processed video...")
        final_output_file = add_audio_from_source(source_path, output_file, final_output, verbose)
        
        if final_output_file:
            print(f"Final video with audio saved to {final_output_file}")
            # Remove the silent video to save space if audio was added successfully
            try:
                os.remove(output_file)
                if verbose:
                    print(f"Removed silent video: {output_file}")
            except Exception as e:
                if verbose:
                    print(f"Could not remove silent video: {str(e)}")
            return final_output_file
        else:
            print(f"Failed to add audio. Silent video is still available at {output_file}")
            return output_file
    
    except Exception as e:
        print(f"Error processing video {source_path}: {str(e)}")
        return None

def add_audio_from_source(source_video, silent_video, output_video, verbose=False):
    """
    Add audio from the source video to the processed silent video.
    
    Args:
        source_video (str): Path to the original video with audio
        silent_video (str): Path to the processed video without audio
        output_video (str): Path to save the final video with audio
        verbose (bool): Whether to show detailed progress information
    
    Returns:
        str: Path to the final video with audio, or None if failed
    """
    try:
        if verbose:
            print(f"Adding audio from {source_video} to {silent_video}")
        
        # Get the audio from the original video
        input_audio = ffmpeg.input(source_video).audio
        
        # Get the video from the processed file
        input_video = ffmpeg.input(silent_video).video
        
        # Combine audio and video
        output = ffmpeg.output(input_video, input_audio, output_video, codec='copy')
        
        # Run the FFmpeg command
        ffmpeg.run(output, quiet=not verbose, overwrite_output=True)
        
        if verbose:
            print(f"Successfully added audio to {output_video}")
        
        return output_video
    except Exception as e:
        print(f"Error adding audio: {str(e)}")
        # If using FFmpeg directly is necessary as a fallback
        try:
            if verbose:
                print("Attempting to use direct FFmpeg command...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', silent_video,
                '-i', source_video,
                '-c:v', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_video
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE if not verbose else None,
                                   stderr=subprocess.PIPE if not verbose else None)
            
            if result.returncode == 0:
                if verbose:
                    print(f"Successfully added audio using direct FFmpeg command")
                return output_video
            else:
                print(f"Error executing FFmpeg command: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                return None
        except Exception as sub_e:
            print(f"Error using direct FFmpeg command: {str(sub_e)}")
            return None

def process_batch(config, verbose=False):
    """Process a batch of videos based on the config."""
    source_path = config['source']
    
    # Check if source is a directory
    if os.path.isdir(source_path):
        # Process all video files in the directory
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        processed_files = []
        
        # Count the number of video files to process
        video_files = [f for f in os.listdir(source_path) 
                      if os.path.isfile(os.path.join(source_path, f)) 
                      and os.path.splitext(f)[1].lower() in video_extensions]
        
        if not video_files:
            print(f"No video files found in {source_path}")
            return []
        
        print(f"Found {len(video_files)} video file(s) to process")
        
        for i, filename in enumerate(video_files):
            file_path = os.path.join(source_path, filename)
            
            print(f"\nProcessing file {i+1}/{len(video_files)}: {filename}")
            # Create a copy of the config with the current file as source
            file_config = config.copy()
            file_config['source'] = file_path
            
            # Process the video
            output_file = process_video(file_config, verbose)
            if output_file:
                processed_files.append(output_file)
        
        return processed_files
    else:
        # Process a single video file
        output_file = process_video(config, verbose)
        return [output_file] if output_file else []

def main():
    """Main function to process videos based on config."""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Video panning effect processor')
        parser.add_argument('--config', required=True, help='Path to configuration JSON file')
        parser.add_argument('--verbose', action='store_true', help='Show detailed progress information')
        args = parser.parse_args()
        
        if not os.path.exists(args.config):
            print(f"Error: Config file not found at {args.config}")
            sys.exit(1)
        
        # Ensure templates directory exists
        ensure_templates_directory()
        
        # Load the configuration
        config = load_config(args.config)
        
        print(f"Starting video processing with configuration from {args.config}")
        
        start_time = time.time()
        
        # Process the video(s)
        output_files = process_batch(config, args.verbose)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if output_files:
            print(f"\nProcessing complete. {len(output_files)} file(s) processed in {elapsed_time:.2f} seconds.")
            for file in output_files:
                print(f"- {file}")
        else:
            print(f"\nNo files were successfully processed.")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
