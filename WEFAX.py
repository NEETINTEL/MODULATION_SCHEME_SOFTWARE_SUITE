#!/usr/bin/env python3
"""
WEFAX (Weather Facsimile) Encoder and Decoder
Supports IOC 576/288 standard with 120/240 RPM
"""

import numpy as np
import wave
import struct
import sys
import os
from typing import Optional, Tuple, Union

# Check if PIL is available for text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: PIL/Pillow NOT INSTALLED. TEXT ENCODING DISABLED.")
    print("INSTALL WITH: pip install Pillow\n")


def get_script_directory() -> str:
    """Get the directory where the script is located"""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_output_directory(subdirectory: Optional[str] = None) -> str:
    """Ensure output directory exists, create if needed"""
    script_dir = get_script_directory()
    output_dir = os.path.join(script_dir, "output", subdirectory) if subdirectory else os.path.join(script_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"CREATED OUTPUT DIRECTORY: {output_dir}\n")
    return output_dir


def get_relative_path(path: str) -> str:
    """Get path relative to script directory"""
    return os.path.relpath(path, get_script_directory())


def get_input(prompt: str, default: str, validator=None) -> str:
    """Generic input handler with validation"""
    while True:
        try:
            value = input(prompt).strip() or default
            if validator and not validator(value):
                continue
            return value
        except KeyboardInterrupt:
            print("\n\nEXITING...")
            sys.exit(0)


class WEFAXCodec:
    """WEFAX (Weather Facsimile) Encoder and Decoder"""
    
    def __init__(self, sample_rate: int = 48000, ioc: int = 576, rpm: int = 120):
        self.sample_rate = sample_rate
        self.ioc = ioc
        self.rpm = rpm
        self.freq_black = 1500
        self.freq_white = 2300
        self.line_duration = 60.0 / self.rpm
    
    def bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, sample_rate: int) -> np.ndarray:
        """Apply bandpass FIR filter using windowed sinc method"""
        transition_width = 500
        num_taps = int(4 * sample_rate / transition_width)
        if num_taps % 2 == 0:
            num_taps += 1
        
        highcut_normalized = highcut / sample_rate
        lowcut_normalized = lowcut / sample_rate
        
        sinc_kernel_lp = np.sinc(2 * highcut_normalized * (np.arange(num_taps) - (num_taps - 1) / 2))
        sinc_kernel_hp_base = np.sinc(2 * lowcut_normalized * (np.arange(num_taps) - (num_taps - 1) / 2))
        sinc_kernel_hp = -sinc_kernel_hp_base
        sinc_kernel_hp[num_taps // 2] += 1
        
        sinc_kernel = sinc_kernel_lp * sinc_kernel_hp * np.blackman(num_taps)
        sinc_kernel = sinc_kernel / np.sum(np.abs(sinc_kernel))
        
        return np.convolve(data, sinc_kernel, mode='same').astype(np.float32)
    
    def sharpen_image(self, pixels: np.ndarray, width: int, height: int) -> np.ndarray:
        """Apply unsharp mask for enhanced sharpness"""
        blurred = np.copy(pixels)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                blurred[y, x] = (
                    pixels[y-1, x-1] + pixels[y-1, x] + pixels[y-1, x+1] +
                    pixels[y, x-1] + pixels[y, x] + pixels[y, x+1] +
                    pixels[y+1, x-1] + pixels[y+1, x] + pixels[y+1, x+1]
                ) / 9.0
        return pixels + 0.6 * (pixels - blurred)
    
    def render_text_to_image(self, text: str, font_size: int = 12, chars_per_line: int = 80) -> Image.Image:
        """Render text as a monochrome image using monospace font"""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL/Pillow IS REQUIRED FOR TEXT RENDERING")
        
        image_width = int(self.ioc * np.pi)
        has_japanese = any(ord(char) > 0x3000 for char in text)
        
        font_names = ["NotoSansMonoCJKhk-Regular.otf", "Hiragino Sans GB.ttc", "AppleGothic.ttf"] if has_japanese else ["IBMPlexMono-Regular.otf", "Courier New.ttf"]
        
        font = None
        for font_name in font_names:
            try:
                font = ImageFont.truetype(font_name, font_size)
                print(f"  USING FONT: {font_name}")
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
            print("  WARNING: USING DEFAULT FONT")
        
        lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                lines.append('')
            else:
                for i in range(0, len(paragraph), chars_per_line):
                    lines.append(paragraph[i:i + chars_per_line])
        
        line_height = font_size + 4
        img = Image.new('L', (image_width, len(lines) * line_height + 20), color=255)
        draw = ImageDraw.Draw(img)
        
        for i, line in enumerate(lines):
            draw.text((10, 10 + i * line_height), line, font=font, fill=0)
        
        return img
    
    def read_bmp(self, filepath: str) -> Tuple[np.ndarray, int, int]:
        """Read a grayscale BMP file and return pixel array"""
        with open(filepath, 'rb') as f:
            header = f.read(54)
            width, height = struct.unpack('<I', header[18:22])[0], struct.unpack('<I', header[22:26])[0]
            bits_per_pixel = struct.unpack('<H', header[28:30])[0]
            
            if bits_per_pixel != 8:
                raise ValueError(f"ONLY 8-BIT GRAYSCALE BMPS SUPPORTED (GOT {bits_per_pixel}-BIT)")
            
            f.seek(54 + 256 * 4)
            row_size = ((width * 8 + 31) // 32) * 4
            pixel_data = [list(f.read(row_size)[:width]) for _ in range(height)]
            pixel_data.reverse()
            
            return np.array(pixel_data, dtype=np.uint8), width, height
    
    def encode_image(self, image: Union[Image.Image, np.ndarray], output_wav: str) -> None:
        """Encode a PIL Image or numpy array to WEFAX WAV audio"""
        if isinstance(image, np.ndarray):
            pixels = image
            height, width = pixels.shape
        else:
            if image.mode != 'L':
                image = image.convert('L')
            pixels = np.array(image, dtype=np.uint8)
            height, width = pixels.shape
        
        print(f"  IMAGE SIZE: {width}x{height} PIXELS")
        print(f"  ENCODING: {self.rpm} RPM, IOC {self.ioc}\n")
        
        samples_per_line = int(self.line_duration * self.sample_rate)
        audio_data = np.zeros(samples_per_line * height, dtype=np.float32)
        
        print(f"ENCODING {height} LINES...")
        
        phase = 0.0
        sample_idx = 0
        
        for y in range(height):
            if y % 10 == 0 or y == height - 1:
                sys.stdout.write(f"\r  LINE {y + 1}/{height}...")
                sys.stdout.flush()
            
            samples_per_pixel = samples_per_line / width
            
            for x in range(width):
                freq = self.freq_black + (pixels[y, x] / 255.0) * (self.freq_white - self.freq_black)
                pixel_end_sample = int((x + 1) * samples_per_pixel)
                num_samples = pixel_end_sample - (sample_idx % samples_per_line)
                phase_increment = 2 * np.pi * freq / self.sample_rate
                
                for _ in range(num_samples):
                    audio_data[sample_idx] = np.sin(phase)
                    phase += phase_increment
                    sample_idx += 1
                
                phase = phase % (2 * np.pi)
        
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        
        print("APPLYING BANDPASS FILTER...")
        audio_data = self.bandpass_filter(audio_data, lowcut=100, highcut=4800, sample_rate=self.sample_rate)
        
        print("NORMALIZING AUDIO...")
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
        
        audio_data_int16 = np.int16(audio_data * 32767)
        
        print("WRITING WAV FILE...")
        with wave.open(output_wav, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data_int16.tobytes())
        
        print("\nENCODING COMPLETE!")
        print(f"  OUTPUT FILE: {get_relative_path(output_wav).upper()}")
        print(f"  DURATION: {int(len(audio_data_int16)) / self.sample_rate:.1f} SECONDS\n")
    
    def read_wav(self, filepath: str, channel: str = 'L') -> Tuple[np.ndarray, int]:
        """Read WAV file with fallback for problematic formats"""
        with open(filepath, 'rb') as f:
            riff = f.read(12)
            if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
                raise ValueError("NOT A VALID WAV FILE")
            
            # Find fmt chunk
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    raise ValueError("FMT CHUNK NOT FOUND")
                
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
                
                if chunk_id == b'fmt ':
                    fmt_data = f.read(chunk_size)
                    audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                    n_channels = struct.unpack('<H', fmt_data[2:4])[0]
                    sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                    bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                    print(f"  FORMAT: {bits_per_sample}-BIT, {n_channels} CH, {sample_rate} HZ")
                    break
                else:
                    f.seek(chunk_size, 1)
            
            # Find data chunk
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    raise ValueError("DATA CHUNK NOT FOUND")
                
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
                
                if chunk_id == b'data':
                    raw_data = f.read(chunk_size)
                    break
                else:
                    f.seek(chunk_size, 1)
            
            # Parse audio data
            if audio_format == 1:  # PCM
                if bits_per_sample == 16:
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                elif bits_per_sample == 8:
                    audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) - 128) * 256
                elif bits_per_sample == 24:
                    audio_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
                    audio = np.zeros(len(audio_bytes), dtype=np.int16)
                    for i in range(len(audio_bytes)):
                        val = audio_bytes[i][0] | (audio_bytes[i][1] << 8) | (audio_bytes[i][2] << 16)
                        if val & 0x800000:
                            val |= 0xFF000000
                        audio[i] = (val // 256)
                else:
                    audio = np.frombuffer(raw_data, dtype=np.int32) // 65536
            elif audio_format == 3:  # IEEE float
                audio_float = np.frombuffer(raw_data, dtype=np.float32)
                audio = (audio_float * 32767).astype(np.int16)
            else:
                raise ValueError(f"UNSUPPORTED AUDIO FORMAT: {audio_format}")
            
            # Handle multichannel
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
                channel_idx = 1 if channel.upper() == 'R' else 0
                audio = audio[:, channel_idx]
                print(f"  USING {'RIGHT' if channel.upper() == 'R' else 'LEFT'} CHANNEL")
            
            return audio.astype(np.float32) / 32768.0, sample_rate
    
    def decode(self, input_wav: str, output_bmp: str, width: Optional[int] = None, 
               height: Optional[int] = None, sharpen: bool = False, channel: str = 'L') -> None:
        """Decode WEFAX WAV audio to BMP image"""
        print("READING WAV FILE...")
        
        audio_data, sample_rate = self.read_wav(input_wav, channel)
        print(f"  AUDIO: {len(audio_data)} SAMPLES\n")
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            print(f"RESAMPLING FROM {sample_rate} HZ TO {self.sample_rate} HZ...")
            duration = len(audio_data) / sample_rate
            new_length = int(duration * self.sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data) - 1, new_length), 
                np.arange(len(audio_data)), 
                audio_data
            ).astype(np.float32)
            sample_rate = self.sample_rate
            print()
        
        samples_per_line = int((60.0 / self.rpm) * sample_rate)
        
        if width is None:
            width = int(self.ioc * np.pi)
            print(f"AUTO-DETECTED WIDTH FROM IOC {self.ioc}: {width} PIXELS")
        
        if height is None:
            height = int(len(audio_data) / samples_per_line)
            print(f"AUTO-DETECTED HEIGHT FROM DURATION: {height} LINES")
        
        print(f"DECODING TO {width}x{height} PIXELS\n")
        
        pixels = np.zeros((height, width), dtype=np.float32)
        samples_per_pixel = samples_per_line / width
        analysis_window = max(int(samples_per_pixel * 1.0), 96)
        fft_size = max(analysis_window * 8, 4096)
        
        # Pre-calculate FFT parameters (moved outside loop for efficiency)
        freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
        valid_range = (freqs >= self.freq_black - 50) & (freqs <= self.freq_white + 50)
        valid_freqs = freqs[valid_range]
        
        print(f"DECODING {height} LINES...")
        
        for y in range(height):
            if y % 10 == 0 or y == height - 1:
                sys.stdout.write(f"\r  LINE {y + 1}/{height}...")
                sys.stdout.flush()
            
            line_start = int(y * samples_per_line)
            
            for x in range(width):
                pixel_start = int(line_start + x * samples_per_pixel)
                window_start = max(0, pixel_start)
                window_end = min(len(audio_data), window_start + analysis_window)
                
                if window_end - window_start < analysis_window // 2:
                    pixels[y, x] = 128
                    continue
                
                signal = audio_data[window_start:window_end]
                if len(signal) < analysis_window:
                    signal = np.pad(signal, (0, analysis_window - len(signal)))
                else:
                    signal = signal[:analysis_window]
                
                signal = signal * np.blackman(analysis_window)
                fft_result = np.fft.rfft(np.pad(signal, (0, fft_size - len(signal))))
                valid_magnitudes = np.abs(fft_result)[valid_range]
                
                if len(valid_magnitudes) > 0 and np.max(valid_magnitudes) > 0.005:
                    peak_idx = np.argmax(valid_magnitudes)
                    
                    if 0 < peak_idx < len(valid_magnitudes) - 1:
                        alpha, beta, gamma = valid_magnitudes[peak_idx - 1:peak_idx + 2]
                        denom = alpha - 2*beta + gamma
                        if abs(denom) > 1e-10:
                            p = 0.5 * (alpha - gamma) / denom
                            detected_freq = valid_freqs[peak_idx] + p * (valid_freqs[1] - valid_freqs[0])
                        else:
                            detected_freq = valid_freqs[peak_idx]
                    else:
                        detected_freq = valid_freqs[peak_idx]
                    
                    intensity = np.clip((detected_freq - self.freq_black) / (self.freq_white - self.freq_black), 0, 1)
                    pixels[y, x] = intensity * 255.0
                else:
                    pixels[y, x] = 128
        
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        
        if sharpen:
            print("APPLYING SHARPENING...")
            pixels = self.sharpen_image(pixels, width, height)
        
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        
        print("WRITING BMP FILE...")
        self.write_bmp(output_bmp, pixels, width, height)
        
        print("\nDECODING COMPLETE!")
        print(f"  OUTPUT FILE: {get_relative_path(output_bmp).upper()}")
        print(f"  IMAGE SIZE: {width}x{height} PIXELS\n")
    
    def write_bmp(self, filepath: str, pixels: np.ndarray, width: int, height: int) -> None:
        """Write a grayscale BMP file"""
        with open(filepath, 'wb') as f:
            row_size = ((width * 8 + 31) // 32) * 4
            pixel_data_size = row_size * height
            file_size = 54 + 1024 + pixel_data_size
            
            f.write(b'BM' + struct.pack('<I', file_size) + struct.pack('<I', 0) + struct.pack('<I', 1078))
            f.write(struct.pack('<I', 40) + struct.pack('<i', width) + struct.pack('<i', height))
            f.write(struct.pack('<H', 1) + struct.pack('<H', 8) + struct.pack('<I', 0))
            f.write(struct.pack('<I', pixel_data_size) + struct.pack('<i', 2835) * 2)
            f.write(struct.pack('<I', 256) * 2)
            
            for i in range(256):
                f.write(struct.pack('BBBB', i, i, i, 0))
            
            for y in range(height - 1, -1, -1):
                f.write(pixels[y].tobytes() + b'\x00' * (row_size - width))


def select_file(file_type: str = 'bmp') -> Tuple[Optional[str], Optional[str]]:
    """Find and let user select a file"""
    extensions = {
        'bmp': (['.bmp'], "BMP"),
        'wav': (['.wav'], "WAV"),
        'text': (['.txt'], "TEXT")
    }
    
    exts, type_name = extensions.get(file_type, (['.bmp'], "BMP"))
    script_dir = get_script_directory()
    
    print(f"SEARCHING FOR {type_name} FILES IN: {script_dir}\n")
    
    files_with_paths = []
    for root, _, files in os.walk(script_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                location = "." if root == script_dir else os.path.relpath(root, script_dir)
                files_with_paths.append((f, root, location))
    
    if not files_with_paths:
        print(f"NO {type_name} FILES FOUND!")
        return None, None
    
    files_with_paths.sort(key=lambda x: x[0].lower())
    
    print(f"AVAILABLE {type_name} FILES:")
    for i, (filename, filepath, location) in enumerate(files_with_paths, 1):
        size_kb = os.path.getsize(os.path.join(filepath, filename)) / 1024
        path_display = filename if location == "." else f"{location}/{filename}"
        print(f"{i:2d}. {path_display} ({size_kb:.1f} KB)")
    
    while True:
        try:
            choice = input(f"\nSELECT [1-{len(files_with_paths)}] OR [0] TO EXIT > ").strip()
            if choice == '0':
                print("EXITING...")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(files_with_paths):
                filename, filepath, _ = files_with_paths[choice_num - 1]
                return filename, filepath
            print(f"ENTER 1-{len(files_with_paths)}")
        except ValueError:
            print("ENTER A VALID NUMBER")
        except KeyboardInterrupt:
            print("\n\nEXITING...")
            sys.exit(0)


def get_rpm_choice() -> int:
    """Get RPM selection from user"""
    print("SELECT IOC: [1] IOC576 (120 RPM), [2] IOC288 (240 RPM)")
    while True:
        choice = get_input("SELECTION [1] > ", '1')
        if choice in ['1', '2']:
            return 120 if choice == '1' else 240
        print("ERROR: ENTER 1 OR 2")


def get_int_input(prompt: str, default: int) -> int:
    """Get validated integer input"""
    while True:
        try:
            user_input = input(prompt).strip()
            value = int(user_input) if user_input else default
            if value < 1:
                print("ERROR: VALUE MUST BE >= 1")
                continue
            return value
        except ValueError:
            print("ERROR: ENTER A VALID NUMBER")
        except KeyboardInterrupt:
            print("\n\nEXITING...")
            sys.exit(0)


def handle_operation(operation: str) -> None:
    """Unified handler for all operations"""
    
    if operation == 'encode_text':
        if not PIL_AVAILABLE:
            print("ERROR: PIL/Pillow REQUIRED FOR TEXT ENCODING")
            print("INSTALL WITH: pip install Pillow")
            sys.exit(1)
        
        rpm = get_rpm_choice()
        print()
        font_size = get_int_input("FONT SIZE (PIXELS) [12] > ", 12)
        chars_per_line = get_int_input("CHARACTERS PER LINE [80] > ", 80)
        print()
        
        filename, filepath = select_file('text')
        if filename is None:
            sys.exit(0)
        
        input_file = os.path.join(filepath, filename)
        print(f"\nSELECTED: {filename}\n")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"ERROR READING FILE: {e}")
            sys.exit(1)
        
        print(f"TEXT LENGTH: {len(text)} CHARACTERS\n")
        
        base_name = os.path.splitext(filename)[0]
        ioc = 576 if rpm == 120 else 288
        output_dir = ensure_output_directory(f"{base_name}_WEFAX_IOC{ioc}")
        output_file = os.path.join(output_dir, f"{base_name}_WEFAX_IOC{ioc}.wav")
        
        print(f"OUTPUT: {get_relative_path(output_file)}")
        
        save_image = input("SAVE RENDERED IMAGE? (Y/N) [Y] > ").strip().upper() != 'N'
        image_file = None
        if save_image:
            image_file = os.path.join(output_dir, f"{base_name}_WEFAX.bmp")
            print(f"IMAGE: {get_relative_path(image_file)}")
        print()
        
        encoder = WEFAXCodec(sample_rate=48000, ioc=576, rpm=rpm)
        print("RENDERING TEXT TO IMAGE...")
        print(f"  FONT SIZE: {font_size}px, CHARS/LINE: {chars_per_line}\n")
        
        image = encoder.render_text_to_image(text, font_size, chars_per_line)
        print(f"  RENDERED {image.width}x{image.height}px IMAGE\n")
        
        if save_image and image_file:
            image.save(image_file, 'BMP')
            print(f"  IMAGE SAVED: {get_relative_path(image_file)}\n")
        
        encoder.encode_image(image, output_file)
    
    elif operation == 'encode_bmp':
        rpm = get_rpm_choice()
        print()
        
        filename, filepath = select_file('bmp')
        if filename is None:
            sys.exit(0)
        
        input_file = os.path.join(filepath, filename)
        print(f"\nSELECTED: {filename}\n")
        
        base_name = os.path.splitext(filename)[0]
        ioc = 576 if rpm == 120 else 288
        output_dir = ensure_output_directory(f"{base_name}_WEFAX_IOC{ioc}")
        output_file = os.path.join(output_dir, f"{base_name}_WEFAX_IOC{ioc}.wav")
        
        print(f"OUTPUT: {get_relative_path(output_file)}\n")
        
        encoder = WEFAXCodec(sample_rate=48000, ioc=576, rpm=rpm)
        print("READING BMP IMAGE...")
        pixels, _, _ = encoder.read_bmp(input_file)
        encoder.encode_image(pixels, output_file)
    
    else:  # decode
        filename, filepath = select_file('wav')
        if filename is None:
            sys.exit(0)
        
        input_file = os.path.join(filepath, filename)
        print(f"\nSELECTED: {filename}\n")
        
        # Auto-detect IOC/RPM from filename
        filename_upper = filename.upper()
        if 'IOC576' in filename_upper or '120RPM' in filename_upper:
            rpm, ioc = 120, 576
            print(f"DETECTED FROM FILENAME: IOC{ioc} (120 RPM)")
        elif 'IOC288' in filename_upper or '240RPM' in filename_upper:
            rpm, ioc = 240, 288
            print(f"DETECTED FROM FILENAME: IOC{ioc} (240 RPM)")
        else:
            rpm = get_rpm_choice()
            ioc = 576 if rpm == 120 else 288
        
        print()
        
        # Check for stereo
        channel = 'L'
        try:
            with open(input_file, 'rb') as f:
                f.read(12)
                while True:
                    chunk_header = f.read(8)
                    if len(chunk_header) < 8:
                        break
                    chunk_id = chunk_header[:4]
                    chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
                    if chunk_id == b'fmt ':
                        fmt_data = f.read(min(chunk_size, 16))
                        n_channels = struct.unpack('<H', fmt_data[2:4])[0]
                        if n_channels > 1:
                            print(f"STEREO AUDIO DETECTED ({n_channels} CH)")
                            channel = get_input("SELECT CHANNEL: [L]EFT OR [R]IGHT [L] > ", 'L')
                            print()
                        break
                    f.seek(chunk_size, 1)
        except:
            pass
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(filepath, f"{base_name}_DECODED.bmp")
        
        print(f"OUTPUT: {get_relative_path(output_file)}\n")
        
        sharpen = input("APPLY SHARPENING? (Y/N) [N] > ").strip().upper() == 'Y'
        print()
        
        decoder = WEFAXCodec(sample_rate=48000, ioc=ioc, rpm=rpm)
        decoder.decode(input_file, output_file, sharpen=sharpen, channel=channel)


def main() -> None:
    """Main entry point"""
    print("=" * 60)
    print("            WEFAX ENCODER/DECODER")
    print("=" * 60)
    print("\nSELECT OPERATION:")
    #print("-" * 60)
    print("  [1] ENCODE BMP TO WAV                    (IMAGE > AUDIO)")
    print("  [2] ENCODE TXT TO WAV             (TEXT > IMAGE > AUDIO)")
    #print("-" * 60)
    print("  [3] DECODE WAV TO BMP                    (AUDIO > IMAGE)")
    #print("-" * 60)
    
    try:
        choice = input("\nSELECT [1], [2], [3], OR INPUT [0] TO EXIT > ").strip().upper()
        if choice == '0':
            print("\nEXITING...")
            sys.exit(0)
        
        if choice not in ['1', '2', '3']:
            print("ERROR: INVALID SELECTION. ENTER 0, 1, 2, OR 3")
            main()
            return
    except KeyboardInterrupt:
        print("\n\nEXITING...")
        sys.exit(0)
    
    print()
    
    operations = {'1': 'encode_bmp', '2': 'encode_text', '3': 'decode'}
    
    try:
        handle_operation(operations[choice])
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
