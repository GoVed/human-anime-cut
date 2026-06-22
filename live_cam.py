import cv2
import numpy as np
import torch
import argparse
import sys
import os
from src.models import Generator

def preprocess_image_for_prediction(image, size, device):
    """Resizes OpenCV RGB image to size x size, normalizes to [-1, 1], and formats to PyTorch tensor."""
    # Resize image
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # Convert to float and scale to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Normalize to [-1, 1]
    img = (img - 0.5) / 0.5
    # Change shape from HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor

def postprocess_prediction(tensor):
    """Converts PyTorch tensor to HWC OpenCV RGB image denormalized to [0, 255]."""
    # Remove batch dimension and move to CPU
    img = tensor.squeeze(0).detach().cpu().numpy()
    # Change shape from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 255]
    img = (img + 1.0) * 127.5
    # Clip values and cast to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def crop_square(image):
    """Crops a center square out of the image."""
    h, w = image.shape[:2]
    if h > w:
        diff = h - w
        return image[diff//2:h-diff//2, :]
    else:
        diff = w - h
        return image[:, diff//2:w-diff//2]

def main():
    parser = argparse.ArgumentParser(description="Live camera scenery style transfer using PyTorch CycleGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to unified CycleGAN training checkpoint (.pt file)")
    parser.add_argument("--image-size", type=int, default=128, help="Image resolution size (e.g., 128, 256, 512)")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g., 0) or video stream URL (e.g., http://192.168.0.245:16500)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    has_reconstruction = False
    style_to_photo_model = None

    # Load models
    print("Loading Generator model...")
    photo_to_style_model = Generator().to(device)

    # Load checkpoint state dicts
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'G_XtoY' in checkpoint and 'G_YtoX' in checkpoint:
            photo_to_style_model.load_state_dict(checkpoint['G_XtoY'])
            style_to_photo_model = Generator().to(device)
            style_to_photo_model.load_state_dict(checkpoint['G_YtoX'])
            has_reconstruction = True
            print("Successfully loaded CycleGAN generator parameters from checkpoint.")
        elif 'G' in checkpoint:
            photo_to_style_model.load_state_dict(checkpoint['G'])
            print("Successfully loaded CUT generator parameters from checkpoint (No reconstruction model).")
        else:
            # Fallback: try loading directly as state dict
            photo_to_style_model.load_state_dict(checkpoint)
            print("Successfully loaded model parameters directly.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    photo_to_style_model.eval()
    if has_reconstruction:
        style_to_photo_model.eval()

    # Open the video source (webcam index or network stream URL)
    video_source = args.source
    try:
        video_source = int(video_source)
    except ValueError:
        pass # Keep as string URL
        
    print(f"Opening video source: {video_source}...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        sys.exit(1)

    print("Live Cam running. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Crop to square
        h, w = frame.shape[:2]
        frame = frame[:, (w-h)//2:(w+h)//2]

        # Flip horizontally
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Crop center square
        frame = crop_square(frame)
        h, w = frame.shape[:2]

        # Display the resulting frame
        try:
            cv2.imshow('Original', frame)

            # Convert BGR to RGB for generator model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess the image
            img_tensor = preprocess_image_for_prediction(frame_rgb, args.image_size, device)
            
            with torch.no_grad():
                # Inference
                style_tensor = photo_to_style_model(img_tensor)
                if has_reconstruction:
                    reconstr_tensor = style_to_photo_model(style_tensor)

            # Postprocess outputs
            style_img = postprocess_prediction(style_tensor)
            if has_reconstruction:
                reconstr_img = postprocess_prediction(reconstr_tensor)

            # Upscale predictions back to input display resolution
            style_img = cv2.resize(style_img, (w, h), interpolation=cv2.INTER_CUBIC)
            if has_reconstruction:
                reconstr_img = cv2.resize(reconstr_img, (w, h), interpolation=cv2.INTER_CUBIC)

            # Convert RGB back to BGR for OpenCV display
            style_img = cv2.cvtColor(style_img, cv2.COLOR_RGB2BGR)    
            if has_reconstruction:
                reconstr_img = cv2.cvtColor(reconstr_img, cv2.COLOR_RGB2BGR)

            cv2.imshow('Anime Style', style_img)
            if has_reconstruction:
                cv2.imshow('Reconstructed Photo', reconstr_img)
            
        except Exception as e:
            print(f"Exception during frame processing: {e}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()