"""Build a multi-size .ico file manually (Pillow's writer is buggy)."""
from PIL import Image
from pathlib import Path
import struct

ASSETS = Path(__file__).parent / "assets"
src = ASSETS / "orca_icon_small.png"
ico = ASSETS / "orca_icon.ico"

img = Image.open(src).convert("RGBA")
print(f"Source: {img.size}")

# Sizes we want in the .ico
target_sizes = [16, 24, 32, 48, 64, 128, 256]

# Resize the image to each size, then encode as PNG bytes (modern .ico supports embedded PNG)
images_data = []
for size in target_sizes:
    resized = img.resize((size, size), Image.LANCZOS)
    from io import BytesIO
    buf = BytesIO()
    resized.save(buf, format="PNG")
    images_data.append((size, buf.getvalue()))

# Build .ico file manually
# Header: reserved(2) + type(2) + count(2) = 6 bytes
# Each entry: width(1) + height(1) + colors(1) + reserved(1) + planes(2) + bpp(2) + size(4) + offset(4) = 16 bytes
# Then the image data

header_size = 6
entry_size = 16
num_images = len(images_data)

# Calculate offsets
current_offset = header_size + entry_size * num_images
entries = []
all_data = b""
for size, png_data in images_data:
    entries.append((size, len(png_data), current_offset))
    all_data += png_data
    current_offset += len(png_data)

# Build header
ico_bytes = struct.pack("<HHH", 0, 1, num_images)

# Build entries
for size, data_size, offset in entries:
    if size >= 256:
        width_byte = 0  # 0 means 256 in ICO format
        height_byte = 0
    else:
        width_byte = size
        height_byte = size
    ico_bytes += struct.pack(
        "<BBBBHHII",
        width_byte, height_byte,  # width, height
        0, 0,                      # color count (0 = no palette), reserved
        1, 32,                     # color planes, bits per pixel
        data_size, offset          # size, offset
    )

# Add image data
ico_bytes += all_data

ico.write_bytes(ico_bytes)
print(f"Created: {ico} ({ico.stat().st_size} bytes)")
print(f"Sizes: {target_sizes}")

# Verify by trying to read with Pillow
try:
    test = Image.open(ico)
    print(f"Pillow read: sizes={test.info.get('sizes')}")
except Exception as e:
    print(f"Pillow read error (file may still be valid): {e}")
