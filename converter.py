import numpy as np
from PIL import Image, UnidentifiedImageError
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class Converter:
    def __init__(self, sprite_path, sprite_width=None, sprite_height=None, sprite_rows=1, sprite_cols=1):
        """
        Initialize the converter with a sprite sheet.
        
        Args:
            sprite_path: Path to the sprite sheet image
            sprite_width: Width of each sprite in the sheet (auto-detected if None)
            sprite_height: Height of each sprite in the sheet (auto-detected if None)
            sprite_rows: Number of rows in the sprite sheet
            sprite_cols: Number of columns in the sprite sheet
        """
        # Validate file format
        self._validate_image_format(sprite_path)
        
        # Open and convert to RGBA
        self.sprite_sheet = Image.open(sprite_path).convert("RGBA")
        self.total_width, self.total_height = self.sprite_sheet.size
        
        # Auto-detect sprite dimensions if not provided
        self.sprite_width = sprite_width or (self.total_width // sprite_cols)
        self.sprite_height = sprite_height or (self.total_height // sprite_rows)
        
        self.sprite_rows = sprite_rows
        self.sprite_cols = sprite_cols
        self.sprites = []
        self.voxel_data = None
        
        # Extract individual sprites
        self._extract_sprites()
        
    def _validate_image_format(self, image_path):
        """
        Validate that the image file is a supported format with transparency.
        
        Args:
            image_path: Path to the image file to validate
            
        Raises:
            ValueError: If the file is not a valid image or doesn't support transparency
            FileNotFoundError: If the file doesn't exist
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if it's a valid image file
        try:
            # Try opening the image to check if it's valid
            with Image.open(image_path) as test_img:
                # Check the format
                img_format = test_img.format
                if not img_format:
                    raise ValueError(f"Unable to determine image format: {image_path}")
                
                # Check for formats that support transparency
                supported_formats = ['PNG', 'GIF', 'WEBP', 'TIFF']
                if img_format not in supported_formats:
                    raise ValueError(
                        f"Image format '{img_format}' doesn't support transparency. "
                        f"Please use one of these formats: {', '.join(supported_formats)}"
                    )
                
                # Verify it's a valid image
                test_img.verify()
                
        except UnidentifiedImageError:
            raise ValueError(f"File is not a valid image: {image_path}")
        except Exception as e:
            raise ValueError(f"Error validating image: {str(e)}")
    
    def _extract_sprites(self):
        """Extract individual sprites from the sprite sheet."""
        for row in range(self.sprite_rows):
            for col in range(self.sprite_cols):
                left = col * self.sprite_width
                upper = row * self.sprite_height
                right = left + self.sprite_width
                lower = upper + self.sprite_height
                
                sprite = self.sprite_sheet.crop((left, upper, right, lower))
                self.sprites.append(sprite)
        
        print(f"Extracted {len(self.sprites)} sprites from sheet.")
    
    def create_voxel_model(self, depth=None, threshold=0):
        """
        Create a voxel model from the sprites.
        
        Args:
            depth: Depth of the resulting model (defaults to sprite count)
            threshold: Alpha threshold for considering a pixel as solid (0-255)
        
        Returns:
            numpy.ndarray: 3D array representing the voxel model
        """
        if not self.sprites:
            raise ValueError("No sprites extracted from sprite sheet.")
        
        # Default depth to sprite count if not specified
        depth = depth or len(self.sprites)
        
        # Initialize 3D array for voxel data
        self.voxel_data = np.zeros((depth, self.sprite_height, self.sprite_width, 4), dtype=np.uint8)
        
        # Map sprites to slices in the voxel model
        for z, sprite in enumerate(self.sprites[:depth]):
            sprite_data = np.array(sprite)
            self.voxel_data[z, :, :, :] = sprite_data
        
        # Create binary voxel representation (1 for solid, 0 for empty)
        self.binary_voxels = (self.voxel_data[:, :, :, 3] > threshold).astype(np.uint8)
        
        print(f"Created voxel model with shape {self.binary_voxels.shape}")
        return self.binary_voxels
    
    def optimize_model(self):
        """
        Remove enclosed voxels that would never be visible.
        """
        if self.binary_voxels is None:
            raise ValueError("Need to create voxel model first")
        
        optimized = np.copy(self.binary_voxels)
        depth, height, width = self.binary_voxels.shape
        
        # For each voxel, check if it's surrounded by other voxels
        for z in range(1, depth-1):
            for y in range(1, height-1):
                for x in range(1, width-1):
                    # If the voxel isn't solid, skip it
                    if self.binary_voxels[z, y, x] == 0:
                        continue
                    
                    # Check if voxel is surrounded on all 6 sides
                    surrounded = (
                        self.binary_voxels[z-1, y, x] and
                        self.binary_voxels[z+1, y, x] and
                        self.binary_voxels[z, y-1, x] and
                        self.binary_voxels[z, y+1, x] and
                        self.binary_voxels[z, y, x-1] and
                        self.binary_voxels[z, y, x+1]
                    )
                    
                    if surrounded:
                        optimized[z, y, x] = 0
        
        removed = np.sum(self.binary_voxels) - np.sum(optimized)
        print(f"Removed {removed} enclosed voxels.")
        self.binary_voxels = optimized
        return optimized
    
    def create_mesh(self):
        """
        Convert voxel data to a mesh using voxel boxes.
        """
        if self.binary_voxels is None:
            raise ValueError("Need to create voxel model first")
        
        depth, height, width = self.binary_voxels.shape
        
        # Initialize an empty mesh
        mesh = trimesh.Trimesh()
        
        # Create individual colored boxes for each voxel
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    # Skip empty voxels
                    if self.binary_voxels[z, y, x] == 0:
                        continue
                    
                    # Create a unit cube
                    box = trimesh.primitives.Box(
                        extents=[1, 1, 1],
                        transform=trimesh.transformations.translation_matrix([x + 0.5, y + 0.5, z + 0.5])
                    )
                    
                    # Get color from original voxel data (normalize from 0-255 to 0-1 range)
                    if hasattr(self, 'voxel_data'):
                        # RGB color (first 3 channels)
                        color = self.voxel_data[z, y, x, 0:3].astype(float) / 255.0
                        # Set face colors (every face has same color)
                        box.visual.face_colors = [color] * len(box.faces)
                    
                    # Append to our combined mesh
                    mesh += box
        
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        return mesh
    
    def save_mesh(self, output_path):
        """
        Save the generated mesh to a file.
        
        Args:
            output_path: Path to save the mesh file (format determined by extension)
        """
        mesh = self.create_mesh()
        mesh.export(output_path)
        print(f"Saved mesh to {output_path}")
    
    def visualize(self):
        """
        Visualize the voxel model with colors.
        """
        if self.binary_voxels is None:
            raise ValueError("Need to create voxel model first")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the dimensions
        depth, height, width = self.binary_voxels.shape
        
        # Prepare data for scatter plot
        x_data, y_data, z_data = [], [], []
        colors = []
        
        # Collect coordinates and colors
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if self.binary_voxels[z, y, x] == 1:
                        x_data.append(x)
                        y_data.append(y)
                        z_data.append(z)
                        
                        # Get color if available
                        if hasattr(self, 'voxel_data'):
                            # Convert RGB to hex color string
                            r, g, b = self.voxel_data[z, y, x, 0:3]
                            colors.append(f'#{r:02x}{g:02x}{b:02x}')
                        else:
                            colors.append('blue')  # Default color
        
        # Use voxel visualization for basic structure
        filled = np.zeros(self.binary_voxels.shape, dtype=bool)
        filled[self.binary_voxels == 1] = True
        
        # Create color array with same shape as binary_voxels
        if hasattr(self, 'voxel_data'):
            # Initialize with transparent
            facecolors = np.zeros(filled.shape + (4,), dtype=float)
            
            # Set colors where voxels exist
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        if self.binary_voxels[z, y, x] == 1:
                            # RGB from voxel data and add alpha
                            facecolors[z, y, x, :3] = self.voxel_data[z, y, x, 0:3] / 255.0
                            facecolors[z, y, x, 3] = 0.7  # Alpha value for semi-transparency
            
            # Plot colored voxels
            ax.voxels(filled, facecolors=facecolors, edgecolor='k')
        else:
            # Plot without colors
            ax.voxels(filled, edgecolor='k', alpha=0.5)
        
        # Set equal aspect ratio
        max_range = max(width, height, depth)
        x_mid = width / 2
        y_mid = height / 2
        z_mid = depth / 2
        
        # Set axis limits
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        plt.show()


# Usage example
if __name__ == "__main__":
    # Example usage with a sprite sheet
    converter = Converter(
        sprite_path="example_sprite_sheet.png",
        sprite_rows=1,  # Adjust based on your sprite sheet layout
        sprite_cols=8   # Adjust based on your sprite sheet layout
    )
    
    # Create and optimize the voxel model
    converter.create_voxel_model()
    converter.optimize_model()
    
    # Save and visualize
    converter.save_mesh("output_model.obj")
    converter.visualize()