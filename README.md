# LavaRand Entropy Extractor
Turn your lava lamp chaos into cryptographic order!

This Python script channels the chaotic beauty of moving lava lamps to generate high-quality, physically-based cryptographic keys.
Inspired by Cloudflare’s LavaRand project (which famously uses a wall of 100+ lava lamps), this tool lets you create your own cryptographically strong keys from the unpredictable flow of lava blobs — right from your webcam or local images!

##### Table of Contents  
[⚙️ What Can This Script Do?](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#%EF%B8%8F-what-can-this-script-do)  
[🔐 Why Use Physical Entropy like Lava Lamps?](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#-why-use-physical-entropy-like-lava-lamps)  
[🚀 How to Use](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#-how-to-use)  
[📦 Installation](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#-installation)  
[🧪 Requirements.txt](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#-requirementstxt)  
[🛠️ Customize & Contribute](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#%EF%B8%8F-customize--contribute)  
[🧙‍♂️ About the Author](https://github.com/MaestroSugo/lavarand-entropy-extractor/tree/main?tab=readme-ov-file#%E2%80%8D%EF%B8%8F-about-the-author)  
<a name="headers"/>

## ⚙️ What Can This Script Do?
Capture a live photo of your lava lamps directly from a webcam, or use an existing local image.
If no webcam or image is available, it automatically generates a synthetic lava lamp image that simulates lava blobs, guaranteeing the script always works.

Thoroughly analyzes the image to extract entropy from raw pixel data, least significant bits, gradients, pixel differences, and texture variances — all combined to maximize randomness.

Performs multi-round cryptographic hashing (SHA-256, SHA-512, and SHA-3 or BLAKE2) to securely mix entropy sources.
Uses PBKDF2 for final key derivation, producing robust cryptographic keys.

Supports generation of:
- AES-128 and AES-256 encryption keys
- Initialization vectors (IV) for AES
- High-quality seeds for RSA key generation
  
Provides an entropy quality analysis tool so you can confirm the randomness strength of your images.

## 🔐 Why Use Physical Entropy like Lava Lamps?
Traditional pseudorandom number generators can be vulnerable to prediction attacks if their entropy sources are poor or exposed. Physical entropy derived from chaotic lava lamp movements is inherently unpredictable, making your cryptographic keys far stronger and more resistant to sophisticated attacks. It’s science and art fused for next-level security.

## 🚀 How to Use
1: Run the script! It will:
    • Try to capture a photo from your webcam.
    • Fall back to a local image if specified.
    • Generate a synthetic lava lamp photo if no live feed or image found.
2: The script analyzes the image and prints cryptographic keys you can use immediately.
3: Use those keys with your favorite cryptographic tools — for example, CyberChef or OpenSSL.

## 📦 Installation
First you'll need to download the repository:
```
git clone https://github.com/MaestroSugo/lavarand-entropy-extractor.git
cd lavarand-entropy-extractor
```

Now you’ll need Python 3 and the following libraries:
- numpy
- pillow
- opencv-python

Install all dependencies at once with:
```
pip install -r requirements.txt
```

Alternatively, to install manually:
```
pip install numpy pillow opencv-python
```

You can now run the script with the following command:
```
python LavaRand_EntropyExtractor.py
```

## 🧪 Requirements.txt
```
numpy
pillow
opencv-python
```

## 🛠️ Customize & Contribute
Feel free to fork, tweak the entropy extraction methods, or improve the synthetic image generator. Your hardware might be different, and creativity is encouraged!

## 🧙‍♂️ About the Author
Made with passion by Giulio "Sugo" Fabbri — Penetration tester, code artisan and cryptography enthusiast.








