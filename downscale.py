import os
from PIL import Image, ImageOps

# Set your input and output paths
input_folder = '/mnt/2tb-1/image_enhancement/experiments/MambaIR_Rareplanes_pretrained/DOTA_split/images/val'
output_hr_folder = '/mnt/2tb-1/image_enhancement/experiments/MambaIR_Rareplanes_pretrained/DOTA_split/images/val_SR_pad/output_cropped_hr'
output_lr_folder = '/mnt/2tb-1/image_enhancement/experiments/MambaIR_Rareplanes_pretrained/DOTA_split/images/val_SR_pad/output_downscaled_lr'

# Create output folders if they don't exist
os.makedirs(output_hr_folder, exist_ok=True)
os.makedirs(output_lr_folder, exist_ok=True)

target_hr_size = (1024, 1024)
target_lr_size = (256, 256)  # Since downscaled by 4

# Supported image types
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(image_extensions):
        continue

    input_path = os.path.join(input_folder, filename)
    with Image.open(input_path) as img:
        img = img.convert("RGB")  # Ensure 3 channels

        # Get original size
        width, height = img.size

        # If image is larger than target, crop it
        crop_w = min(width, target_hr_size[0])
        crop_h = min(height, target_hr_size[1])
        img_cropped = img.crop((0, 0, crop_w, crop_h))

        # Pad image to 1024x1024 (top-left alignment)
        pad_w = target_hr_size[0] - crop_w
        pad_h = target_hr_size[1] - crop_h
        img_padded = ImageOps.expand(img_cropped, border=(0, 0, pad_w, pad_h), fill=0)

        # Save HR version
        hr_save_path = os.path.join(output_hr_folder, filename)
        img_padded.save(hr_save_path)

        # Downscale to LR and save
        lr_img = img_padded.resize(target_lr_size, Image.BILINEAR)
        lr_save_path = os.path.join(output_lr_folder, filename)
        lr_img.save(lr_save_path)

        print(f"Processed: {filename} | HR: {target_hr_size} | LR: {target_lr_size}")

print("✅ All images padded and saved.")