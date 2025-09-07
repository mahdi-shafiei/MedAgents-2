"""Main Figure PDF Display - Figure 0a"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

# Try to import pdf2image, but handle gracefully if not available
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from .plot_utils import get_figure_0_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_figure_0_colors, apply_medagents_style

def plot_main_figure_pdf(ax, pdf_path=None, panel_label='A'):
    """Display the main figure PDF as an image"""
    
    if pdf_path is None:
        # Default path to main_figure.pdf in same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, 'main_figure.pdf')
    
    if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
        print("Warning: pdf2image and/or PIL not available. Using placeholder.")
        _create_placeholder_figure(ax, panel_label)
        return ax
    
    try:
        # Convert PDF to image using pdf2image
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        
        if images:
            # Get the first (and only) page
            pdf_image = images[0]
            
            # Convert PIL image to numpy array for matplotlib
            img_array = np.array(pdf_image)
            
            # Display the image with original aspect ratio
            ax.imshow(img_array, aspect='equal')
            ax.axis('off')  # Remove axes for clean display
            
            # Add panel label to match other panels
            ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=25, fontweight='bold')
            
            print(f"Successfully loaded main figure PDF from {pdf_path}")
        else:
            _create_placeholder_figure(ax, panel_label)
            
    except Exception as e:
        print(f"Warning: Could not load PDF {pdf_path}: {e}")
        _create_placeholder_figure(ax, panel_label)
    
    return ax

def _create_placeholder_figure(ax, panel_label):
    """Create a placeholder when PDF cannot be loaded"""
    ax.text(0.5, 0.5, 'Main Figure\n(PDF Display)', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add panel label to match other panels
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_0_colors()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_main_figure_pdf(ax, panel_label='A')
    plt.tight_layout()
    plt.savefig('main_figure_display_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()