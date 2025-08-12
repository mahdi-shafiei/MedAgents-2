import matplotlib.pyplot as plt
import numpy as np

def set_plot_style():
    """Set the standard plot style for MedAgents figures."""
    plt.style.use('medagents.mplstyle')

def get_colors(theme='manchester_united'):
    """Get color palette with different themes."""
    if theme == 'vibrant':
        return {
            'red': '#da020e',
            'blue': '#0e2347',
            'green': '#43a047',
            'orange': '#e15007',
            'purple': '#8e24aa',
            'teal': '#00897b',
            'indigo': '#3949ab',
            'pink': '#e91e63',
            'amber': '#ffe500',
            'lime': '#cddc39'
        }
    elif theme == 'manchester_united_official':
        return {
            "black":               "#000000",
            "jet":                 "#444444",
            "gray":                "#888888",
            "french_gray":         "#BCB9BE",
            "snow":                "#FFF0F7",
            "penn_red":            "#BB3234",
            "barn_red":            "#A40000",
            "engineering_orange":  "#D26367",
            "pink":                "#FFC5CE",
            "jasmine":             "#FFE697",
            "sunset":              "#F9C578",
            "sandy_brown":         "#F2A358", 
            "sandy_brown_light":   "#F8D1AB",
            "persimmon":           "#F47B3C",
        }
    elif theme == 'manchester_united':
        return {
            'red_light': '#ffe4e9',
            'red_medium': '#ffc5ce',
            'red_dark': '#d32f2f',
            'black': '#0d0d0d',
            'gold_light': '#ffe697',
            'gold_medium': '#ffd23f',
            'gold_dark': '#f7931e',
            'white': '#fff7f7',
            'gray_light': '#f5f5f5',
            'gray_medium': '#9e9e9e',
            'gray_dark': '#424242',
            'blue_light': '#e3f2fd',
            'blue_medium': '#90caf9',
            'blue_dark': '#1565c0',
            'green_light': '#e8f5e9',
            'green_medium': '#81c784',
            'green_dark': '#1b5e20'
        }
    elif theme == 'pnas':
        return {
            'black': '#000000',
            'blue': '#0072B2',
            'orange': '#E69F00',
            'sky': '#56B4E9',
            'green': '#009E73',
            'yellow': '#F0E442',
            'red': '#D55E00',
            'purple': '#CC79A7',
            'gray': '#999999'
        }
    else:
        return get_colors('pnas')

def get_color_scheme(figure_type='figure_2', theme='manchester_united_official'):
    """Get color scheme for different figure types."""
    base = 'manchester_united_official' if theme is None else theme
    colors = get_colors(base)
    
    color_schemes = {
        'manchester_united_official': {
            'figure_0': {
                'palette': colors,
                'methods': {
                    'EBAgents': colors['jasmine'],
                    'AFlow': colors['sandy_brown_light'],
                    'SPO': colors['sunset'],
                    'MultiPersona': colors['sandy_brown'],
                    'Self-refine': colors['persimmon'],
                    'CoT': colors['barn_red'],
                    'CoT-SC': colors['penn_red'],
                    'MedAgents': colors['pink'],
                    'MDAgents': colors['snow'],
                    'MedPrompt': colors['french_gray'],
                    'MedRAG': colors['gray'],
                    'Few-shot': colors['jet'],
                    'Zero-shot': colors['black']
                },
                'metrics': {
                    'accuracy': colors['penn_red'],
                    'time': colors['jasmine'],
                    'cost': colors['black']
                },
                'annotations': {
                    'text_box_face': colors['snow'],
                    'text_box_edge': colors['gray'],
                    'text': colors['black']
                },
                'legend': {
                    'neutral_marker': colors['gray']
                }
            },
            'figure_1': {
                'palette': colors,
                'no_search': colors['jet'],
                'with_search': colors['penn_red'],
                'rounds': {1: colors['penn_red'], 2: colors['sandy_brown'], 3: colors['jasmine']},
                'architecture': {
                    'agent': colors['black'],
                    'search': colors['sandy_brown'],
                    'reasoning': colors['sunset'],
                    'knowledge': colors['penn_red']
                },
                'metrics': {
                    'accuracy': colors['penn_red'],
                    'time': colors['sandy_brown'],
                    'cost': colors['black']
                },
                'role_play': {
                    'enable_role_play': colors['penn_red'],
                    'disable_role_play': colors['gray']
                },
                'orchestration': {
                    'group_chat_with_orchestrator': colors['penn_red'],
                    'group_chat_voting_only': colors['sunset'],
                    'independent': colors['gray'],
                    'one_on_one_sync': colors['jasmine']
                },
                'improvements': {
                    'Multi-Agent': colors['penn_red'],
                    'Iterative': colors['sunset'],
                    'Evidence': colors['jasmine'],
                    'Role Play': colors['black'],
                    'Discussion': colors['jet']
                },
                'annotations': {
                    'text_box_face': colors['snow'],
                    'text_box_edge': colors['gray'],
                    'text': colors['black']
                },
                'legend': {
                    'neutral_marker': colors['gray']
                }
            },
            'figure_2': {
                'palette': colors,
                'modality': {
                    'both': colors['penn_red'],
                    'vector_only': colors['jasmine'],
                    'web_only': colors['black']
                },
                'features': {
                    'baseline': colors['penn_red'],
                    'no_document_review': colors['sunset'],
                    'no_query_rewrite': colors['jet'],
                    'no_rewrite_no_review': colors['gray']
                },
                'history': {
                    'individual': colors['penn_red'],
                    'shared': colors['sandy_brown']
                },
                'depth': {
                    'more_docs': colors['penn_red']
                },
                'architecture': {
                    'search_module': colors['sandy_brown'],
                    'vector_search': colors['penn_red'],
                    'web_search': colors['black'],
                    'query_rewrite': colors['sunset'],
                    'document_review': colors['pink']
                },
                'metrics': {
                    'accuracy': colors['penn_red'],
                    'time': colors['jasmine'],
                    'cost': colors['black']
                },
                'annotations': {
                    'text_box_face': colors['snow'],
                    'text_box_edge': colors['gray'],
                    'text': colors['black']
                },
                'legend': {
                    'neutral_marker': colors['gray']
                }
            }
        }
    }
    
    return color_schemes.get(base, {}).get(figure_type, colors)

def get_background_colors():
    """Get background color options."""
    return {
        'white': '#ffffff',
        'light_gray': '#f8f9fa',
        'cream': '#fafafa',
        'light_blue': '#f3f8ff',
        'light_green': '#f1f8e9',
        'light_orange': '#fff3e0',
        'light_purple': '#f3e5f5',
        'light_red': '#ffebee'
    }

def get_sequential_colors(n_colors=5, theme='manchester_united_official'):
    """Get sequential colors for gradients with improved color schemes."""
    if theme == 'vibrant':
        colors = ['#e53935', '#ff6b6b', '#ff8a80', '#ffcdd2', '#ffebee']
    elif theme == 'manchester_united':
        colors = list(get_colors('manchester_united').values())
    elif theme == 'manchester_united_official':
        # Create a more sophisticated sequential color scheme
        base_colors = get_colors('manchester_united_official')
        
        if n_colors <= 5:
            # Use a carefully selected subset for small numbers
            colors = [
                base_colors['penn_red'],      # Strong red
                base_colors['engineering_orange'],  # Orange
                base_colors['sunset'],        # Sunset orange
                base_colors['jasmine'],       # Light yellow
                base_colors['pink']           # Light pink
            ]
        elif n_colors <= 8:
            # Extended palette for medium numbers
            colors = [
                base_colors['penn_red'],      # Strong red
                base_colors['barn_red'],      # Darker red
                base_colors['engineering_orange'],  # Orange
                base_colors['persimmon'],    # Dark orange
                base_colors['sunset'],        # Sunset orange
                base_colors['sandy_brown'],   # Sandy brown
                base_colors['jasmine'],       # Light yellow
                base_colors['pink']           # Light pink
            ]
        else:
            # Generate interpolated colors for larger numbers
            # Start with key colors and interpolate between them
            key_colors = [
                base_colors['penn_red'],
                base_colors['engineering_orange'],
                base_colors['sunset'],
                base_colors['jasmine'],
                base_colors['pink']
            ]
            
            # Use matplotlib's color interpolation
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list('custom', key_colors)
            colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n_colors)]
            
    elif theme == 'pnas':
        colors = ["#DF5C60", "#E37B6D", "#E89E7E", "#EDB186", "#F6C493", "#FFD59B", "#F5B78B", "#EB977C", "#E4786C", "#E26A66"]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    
    return colors[:n_colors]

def get_dataset_colors(theme='manchester_united_official'):
    """Get optimized colors for dataset visualization in expert profiles figure.
    
    This function provides a carefully curated color palette specifically designed
    for distinguishing between different medical datasets in bar charts.
    
    Args:
        theme (str): Color theme to use
        
    Returns:
        dict: Mapping of dataset names to optimized colors
    """
    if theme == 'manchester_united_official':
        base_colors = get_colors('manchester_united_official')
        
        # Define dataset-specific colors with good visual separation
        dataset_colors = {
            'medbullets': base_colors['penn_red'],           # Strong red - primary dataset
            'medexqa': base_colors['engineering_orange'],     # Orange - good contrast
            'medmcqa': base_colors['sunset'],                # Sunset orange - distinct
            'medqa': base_colors['jasmine'],                 # Light yellow - bright
            'medxpertqa-r': base_colors['persimmon'],        # Dark orange - rich
            'medxpertqa-u': base_colors['sandy_brown'],      # Sandy brown - earthy
            'mmlu': base_colors['pink'],                     # Light pink - soft
            'mmlu-pro': base_colors['barn_red'],             # Darker red - deep
            'pubmedqa': base_colors['jet']                   # Dark gray - neutral
        }
        
        # Add fallback colors for any additional datasets
        fallback_colors = [
            base_colors['french_gray'],
            base_colors['gray'],
            base_colors['snow']
        ]
        
        return dataset_colors, fallback_colors
        
    elif theme == 'pnas':
        # PNAS-style dataset colors
        dataset_colors = {
            'medbullets': '#D55E00',      # Red
            'medexqa': '#E69F00',         # Orange
            'medmcqa': '#F0E442',         # Yellow
            'medqa': '#009E73',           # Green
            'medxpertqa-r': '#56B4E9',    # Sky blue
            'medxpertqa-u': '#0072B2',    # Blue
            'mmlu': '#CC79A7',            # Purple
            'mmlu-pro': '#E69F00',        # Orange variant
            'pubmedqa': '#999999'         # Gray
        }
        return dataset_colors, []
        
    else:
        # Default to sequential colors for other themes
        colors = get_sequential_colors(9, theme)
        dataset_names = ['medbullets', 'medexqa', 'medmcqa', 'medqa', 
                        'medxpertqa-r', 'medxpertqa-u', 'mmlu', 'mmlu-pro', 'pubmedqa']
        return dict(zip(dataset_names, colors)), []

def get_categorical_colors(n_colors=6, theme='manchester_united_official'):
    """Get categorical colors for different groups."""
    if theme == 'vibrant':
        colors = ['#e53935', '#1e88e5', '#43a047', '#ff9800', '#8e24aa', '#00897b']
    elif theme == 'manchester_united':
        colors = list(get_colors('manchester_united').values())
    elif theme == 'pnas':
        colors = ["#DF5C60", "#E37B6D", "#E89E7E", "#EDB186", "#F6C493", "#FFD59B", "#F5B78B", "#EB977C", "#E4786C", "#E26A66"]
    elif theme == 'manchester_united_official':
        colors = list(get_colors('manchester_united_official').values())
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
    return colors[:n_colors]

def get_marker_styles():
    """Get marker styles."""
    return {
        'no_search': 'o',
        'with_search': 's',
        'primary': 'o',
        'secondary': 's',
        'accent': '^',
        'diamond': 'D',
        'triangle': 'v'
    }

def get_line_styles():
    """Get line styles."""
    return {
        1: ':',
        2: '--',
        3: '-',
        'solid': '-',
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.'
    }

def apply_standard_plot_formatting(ax, panel_label=None, grid_alpha=0.15, grid_linewidth=0.5, background_color='white', fontsize=10, pad=6):
    """Apply standard formatting to a plot axis with PNAS-style panel labeling."""
    if panel_label is not None:
        label_text = str(panel_label).strip().upper()
        ax.text(-0.02, 1.02, label_text, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=fontsize, fontweight='bold')

    ax.set_facecolor(background_color)
    if grid_alpha and grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linewidth=grid_linewidth, axis='y')

    if ax.get_xaxis() is not None and ax.xaxis.label is not None:
        ax.xaxis.label.set_weight('bold')
    if ax.get_yaxis() is not None and ax.yaxis.label is not None:
        ax.yaxis.label.set_weight('bold')

def get_standard_figure_size():
    """Get default standard figure size in inches (two-column width)."""
    return (7.0, 4.5)

def get_pnas_figure_size(layout='two_column'):
    """Get PNAS-conformant figure sizes in inches.

    layout can be one of: 'single_column', 'one_and_half', 'two_column', 'full_page'.
    """
    if layout == 'single_column':
        return (3.43, 2.6)
    if layout == 'one_and_half':
        return (4.57, 3.4)
    if layout == 'full_page':
        return (7.5, 9.0)
    return (7.0, 4.5)

def get_standard_gridspec_params():
    """Get standard gridspec parameters."""
    return {
        'height_ratios': [1, 1],
        'width_ratios': [1.2, 1, 1],
        'hspace': 0.35,
        'wspace': 0.4
    }

def print_color_palette():
    """Print available colors."""
    print("=== Color Palettes ===")
    for theme in ['vibrant', 'manchester_united']:
        print(f"\n{theme.upper()}:")
        colors = get_colors(theme)
        for name, color in colors.items():
            print(f"  {name}: {color}")
    
    print("\n=== Background Colors ===")
    backgrounds = get_background_colors()
    for name, color in backgrounds.items():
        print(f"  {name}: {color}")

# Backward compatibility aliases
get_figure_1_color_scheme = lambda: get_color_scheme('figure_1')
get_figure_2_color_scheme = lambda: get_color_scheme('figure_2') 