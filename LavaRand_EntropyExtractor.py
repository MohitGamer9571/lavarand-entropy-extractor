#!/usr/bin/env python3
"""
LAVARAND ENTROPY EXTRACTOR
===========================

Python script to generate cryptographic keys from lava lamp images
using advanced physical entropy extraction techniques.


Based on Cloudflare's LavaRand system, this script:
1. Capture a photo from the webcam
2. Extracts entropy from multiple sources (pixels, LSB, gradients, variance)
3. Applies multi-round cryptographic mixing (SHA-256, SHA-512, SHA-3/BLAKE2)
4. Uses PBKDF2 for final key derivation
5. Returns a cryptographic key of high quality


Author: Giulio "Sugo" Fabbri
Date: September 2025
"""

import hashlib
import numpy as np
from PIL import Image
import os
import secrets
import struct
import cv2

# GLOBAL FLAG to show internal debug prints the first time only
show_internal_prints = True


def capture_lava_lamp_image(camera_index=0, output_path='lava_lamp_capture.jpg'):
    """
    Capture a single photo from the webcam and save it
    
    Args:
        camera_index (int): Camera ID (default 0)
        output_path (str): File path to save the photo
    
    Returns:
        str: Path to saved image
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture image")

    cv2.imwrite(output_path, frame)
    cap.release()
    return output_path


lava_lamp_photo = capture_lava_lamp_image()
#lava_lamp_photo = 'lava_lamp_photo.jpg'
"""
Remove the comment above if you want to use a local photo instead of the webcam capture.
"""


def lava_lamp_entropy_extractor(image_path, key_length=32, output_format='hex'):
    """
    Extracts entropy from a lava lamp image and generates a cryptographic key

    Args:
        image_path (str): Path to the lava lamp image
        key_length (int): Key length in bytes (32 = 256 bits, 16 = 128 bits)
        output_format (str): 'hex', 'bytes', or 'base64'

    Returns:
        str or bytes: Generated cryptographic key
    """
    global show_internal_prints
    try:
        # Load the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to numpy array for processing
            img_array = np.array(img)

        if show_internal_prints:
            print(f"   Image loaded: {img_array.shape}")

        # 1. MULTI-LEVEL ENTROPY EXTRACTION
        entropy_sources = []

        # a) All pixel values (RGB)
        all_pixels = img_array.flatten()
        entropy_sources.append(all_pixels.tobytes())
        if show_internal_prints:
            print(f"   Total pixels extracted: {len(all_pixels)}")

        # b) Least Significant Bits (LSB) - high randomness
        lsb_bits = []
        for channel in range(3):  # R, G, B
            channel_data = img_array[:, :, channel]
            # Extract last 2 bits of each pixel (most random)
            lsb = channel_data & 0x03  # Mask for last 2 bits
            lsb_bits.extend(lsb.flatten())

        lsb_array = np.array(lsb_bits, dtype=np.uint8)
        entropy_sources.append(lsb_array.tobytes())
        if show_internal_prints:
            print(f"   LSB extracted: {len(lsb_bits)}")

        # c) Differences between adjacent pixels (captures variation)
        diff_horizontal = np.diff(img_array, axis=1)
        diff_vertical = np.diff(img_array, axis=0)
        entropy_sources.append(diff_horizontal.tobytes())
        entropy_sources.append(diff_vertical.tobytes())

        # d) Gradients and edge detection
        gray = np.mean(img_array, axis=2).astype(np.uint8)

        # Sobel gradient
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Simplified manual convolution
        grad_data = []
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                window = gray[i - 1:i + 2, j - 1:j + 2]
                gx = np.sum(window * sobel_x)
                gy = np.sum(window * sobel_y)
                grad_data.extend([gx & 0xFF, gy & 0xFF])  # Only keep 8 bits

        entropy_sources.append(bytes(grad_data))
        if show_internal_prints:
            print(f"   Gradients extracted: {len(grad_data)}")

        # e) Local variances (captures texture)
        block_size = 8
        variance_data = []
        for i in range(0, gray.shape[0] - block_size, block_size):
            for j in range(0, gray.shape[1] - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                var = np.var(block)
                # Convert variance to bytes
                var_bytes = struct.pack('f', var)
                variance_data.extend(var_bytes)

        entropy_sources.append(bytes(variance_data))
        if show_internal_prints:
            print(f"   Local variances extracted: {len(variance_data)} bytes")

        # 2. MULTI-ROUND CRYPTOGRAPHIC MIXING
        # Combine all entropy sources
        combined_entropy = b''.join(entropy_sources)
        if show_internal_prints:
            print(f"   Total combined entropy: {len(combined_entropy)} bytes")

        # Multiple rounds of hashing for optimal mixing
        current_hash = combined_entropy

        # Round 1: SHA-256
        hasher1 = hashlib.sha256()
        hasher1.update(current_hash)
        hash1 = hasher1.digest()

        # Round 2: SHA-512 plus extra salt
        hasher2 = hashlib.sha512()
        hasher2.update(hash1)
        hasher2.update(combined_entropy[-256:])  # Last 256 bytes as salt
        hash2 = hasher2.digest()

        # Round 3: SHA-3 (if available) else BLAKE2
        try:
            hasher3 = hashlib.sha3_256()
            hasher3.update(hash2)
            hasher3.update(current_hash[:256])  # First 256 bytes as salt
            final_hash = hasher3.digest()
        except AttributeError:
            # Fallback to BLAKE2 if SHA-3 not available
            hasher3 = hashlib.blake2b(digest_size=32)
            hasher3.update(hash2)
            hasher3.update(current_hash[:256])
            final_hash = hasher3.digest()

        # 3. KEY DERIVATION FUNCTION (KDF)
        # Final key stretching with PBKDF2
        salt = hash1[:16]  # Use part of first hash as salt
        iterations = 100000  # Large number of iterations

        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            final_hash,
            salt,
            iterations,
            dklen=key_length
        )

        # Disable internal prints after first run
        show_internal_prints = False

        # 4. OUTPUT FORMATTING
        if output_format == 'hex':
            return derived_key.hex()
        elif output_format == 'base64':
            import base64
            return base64.b64encode(derived_key).decode()
        else:
            return derived_key

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


def analyze_entropy_quality(image_path):
    """
    Analyzes the entropy quality in the image
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)

        # Calculate basic statistics
        all_pixels = img_array.flatten()

        stats = {
            'total_pixels': len(all_pixels),
            'unique_values': len(np.unique(all_pixels)),
            'mean': np.mean(all_pixels),
            'std_dev': np.std(all_pixels),
            'min_val': np.min(all_pixels),
            'max_val': np.max(all_pixels),
            'entropy_estimate': 0
        }

        # Estimate Shannon entropy
        unique, counts = np.unique(all_pixels, return_counts=True)
        probabilities = counts / len(all_pixels)
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        stats['entropy_estimate'] = shannon_entropy

        # Simple randomness test
        # Count 0->1 transitions in the LSB
        lsb = all_pixels & 1
        transitions = np.sum(np.diff(lsb) != 0)
        stats['lsb_transitions'] = transitions
        stats['lsb_transition_ratio'] = transitions / (len(lsb) - 1)

        return stats

    except Exception as e:
        print(f"Error during analysis: {e}")
        return None


def create_test_image():
    """
    Creates a test image simulating a lava lamp pattern
    """
    # Generates a test image with colored circular patterns (simulated blobs)
    np.random.seed()

    width, height = 400, 300
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Add random circular blobs (simulate lava blobs)
    for _ in range(20):
        center_x = np.random.randint(50, width - 50)
        center_y = np.random.randint(50, height - 50)
        radius = np.random.randint(20, 60)
        color = np.random.randint(0, 256, 3)

        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        img_array[mask] = color

    # Add noise to boost entropy
    noise = np.random.normal(0, 20, (height, width, 3))
    img_array = np.clip(img_array.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    # Save the test image
    test_img = Image.fromarray(img_array)
    test_img.save('test_lava_lamp.png')
    print("Test image generated: test_lava_lamp.png")

    return 'test_lava_lamp.png'


# Utility functions for different encryption algorithms
def generate_aes_key(image_path, key_size=256):
    """Generates an AES key (128, 192, or 256 bits)"""
    key_bytes = key_size // 8
    return lava_lamp_entropy_extractor(image_path, key_length=key_bytes, output_format='hex')


def generate_rsa_seed(image_path):
    """Generates a high-quality seed for RSA key generation"""
    return lava_lamp_entropy_extractor(image_path, key_length=64, output_format='hex')


def generate_iv(image_path, algorithm='AES'):
    """Generates an Initialization Vector for encryption algorithms"""
    iv_lengths = {'AES': 16, 'DES': 8, 'ChaCha20': 12}
    length = iv_lengths.get(algorithm, 16)
    return lava_lamp_entropy_extractor(image_path, key_length=length, output_format='hex')


# Full example usage
if __name__ == "__main__":
    print("=== LAVA LAMP ENTROPY EXTRACTOR ===")
    print()

    # Create a test image if a lava lamp photo does not exist
    if not os.path.exists(lava_lamp_photo):
        print("No lava lamp photo found, creating a test image...")
        test_image_path = create_test_image()
        image_path = test_image_path
    else:
        image_path = lava_lamp_photo

    print(f"Using image: {image_path}")
    print()

    # Analyze entropy quality
    print("1. ENTROPY QUALITY ANALYSIS:")
    stats = analyze_entropy_quality(image_path)
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    print()

    # Generate keys for different algorithms
    print("2. CRYPTOGRAPHIC KEY GENERATION:")
    print()

    # Generate keys while showing debug print the first time only
    aes_256_key = generate_aes_key(image_path, 256)
    aes_128_key = generate_aes_key(image_path, 128)
    aes_iv = generate_iv(image_path, 'AES')
    rsa_seed = generate_rsa_seed(image_path)

    # Now disable debug prints for any subsequent calls
    show_internal_prints = False

    # Print all keys clearly at the end
    print("\n========== GENERATED KEYS ==========")
    print(f"AES-256 key: {aes_256_key}")
    print(f"AES-128 key: {aes_128_key}")
    print(f"IV for AES: {aes_iv}")
    print(f"RSA Seed: {rsa_seed}")
    print("======================================")