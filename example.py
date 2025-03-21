from converter import Converter

# Path to your sprite sheet
SPRITE_SHEET_PATH = "target.png"

# Create the converter
# Adjust these parameters to match your sprite sheet layout
converter = Converter(
    sprite_path=SPRITE_SHEET_PATH,
    sprite_rows=4,            # Number of rows in your sprite sheet
    sprite_cols=4,            # Number of columns in your sprite sheet
    sprite_width=32,          # Width of each sprite in pixels (optional)
    sprite_height=32          # Height of each sprite in pixels (optional)
)

# Create the initial voxel model
converter.create_voxel_model(
    depth=None,               # Auto-determine based on sprite count
    threshold=128             # Alpha threshold (0-255) for considering a pixel as solid
)

# Optimize by removing enclosed voxels
converter.optimize_model()

# Save the resulting mesh
converter.save_mesh("target.obj")

# Visualize the result
converter.visualize()
