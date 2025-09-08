"""
Add Vignesh to Certificate
Simple script to add "Vignesh" to certificate with predefined settings.
Uses centralized asset management for proper path handling.
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Add the parent directory to the path to import asset_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from asset_manager import AssetManager

def add_vignesh_to_certificate():
    """Add 'Vignesh' to the certificate with predefined settings."""
    
    # Initialize asset manager
    am = AssetManager()
    
    # Predefined settings
    name = "Vignesh Dhanasekaran"
    x = 200
    y = 620
    font_size = 50
    text_color = "#000000"
    
    # Get paths using asset manager
    certificate_path = am.get_asset_path('comfyui', 'certificates', 'Certificate.jpeg')
    output_path = am.get_asset_path('comfyui', 'certificates', 'Certificate_with_name1.jpeg')
    
    print(f"   Adding '{name}' to certificate...")
    print(f"   Position: ({x}, {y})")
    print(f"   Font size: {font_size}")
    print(f"   Color: {text_color}")
    
    # Load the certificate image
    image = Image.open(certificate_path)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Convert hex color to RGB
    if text_color.startswith('#'):
        text_color = text_color[1:]
    rgb_color = tuple(int(text_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Add text to the image
    draw.text((x, y), name, fill=rgb_color, font=font)
    
    # Save the image
    image.save(output_path, "JPEG", quality=95)
    
    print(f"✅ Successfully added '{name}' to certificate!")
    print(f"📁 Output saved to: {output_path}")
    
    # Register the output asset
    am.register_asset('comfyui', 'certificates', 'Certificate_with_name1.jpeg', 
                     output_path, {'name': name, 'position': (x, y), 'font_size': font_size})
    print("📝 Asset registered in centralized registry")

if __name__ == "__main__":
    add_vignesh_to_certificate()
