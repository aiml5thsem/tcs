from stegano import lsb
import re

# remove non-printable chars
with open('sample.txt', 'r', encoding='utf-8') as f:
    raw_code = f.read()

# Remove problematic chars, keep only printable ASCII
clean_code = re.sub(r'[^\x20-\x7E\n\r\t]', '', raw_code)
print(f"[*] Original: {len(raw_code)} chars → Cleaned: {len(clean_code)} chars")

image_path = 'image.png'
output_path = 'image1.png'

try:
    output_image = lsb.hide(image_path, clean_code)
    output_image.save(output_path)
    print(f"✅ Saved to {output_path}")
except Exception as e:
    print(f"❌ Error: {e}")

