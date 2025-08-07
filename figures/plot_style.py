import matplotlib.pyplot as plt
import numpy as np

def set_plot_style():
    """Set the standard plot style for MedAgents figures"""
    plt.style.use('medagents.mplstyle')

def get_colors(theme='manchester_united'):
    """Get color palette with different themes"""
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
            "snow":                "#FFF0F7",
            "penn_red":            "#BB3234",
            "barn_red":            "#A40000",
            "engineering_orange":  "#D26367",
            "pink":                "#FFC5CE",
            "jasmine":             "#FFE697",
            "sunset":              "#F9C578",
            "sandy_brown":         "#F2A358",
            "persimmon":           "#E45F18",
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
    else:
        return get_colors('vibrant')

def get_color_scheme(figure_type='figure_2'):
    """Get color scheme for different figure types"""
    colors = get_colors('manchester_united')
    
    if figure_type == 'figure_1':
        return {
            'no_search': colors['red_dark'],
            'with_search': colors['gold_dark'],
            'rounds': {1: colors['red_dark'], 2: colors['gold_dark'], 3: colors['gold_dark']},
            'architecture': {
                'agent': colors['black'],
                'search': colors['gold_dark'],
                'reasoning': colors['gold_dark'],
                'knowledge': colors['red_dark']
            }
        }
    elif figure_type == 'figure_2':
        return {
            'modality': {
                'both': colors['red_dark'],
                'vector_only': colors['gold_dark'],
                'web_only': colors['black']
            },
            'features': {
                'baseline': colors['red_dark'],
                'no_document_review': colors['gold_dark'],
                'no_query_rewrite': colors['black'],
                'no_rewrite_no_review': colors['gray_medium']
            },
            'history': {
                'separate': colors['red_dark'],
                'shared': colors['gold_dark']
            },
            'depth': {
                'more_docs': colors['red_dark']
            },
            'architecture': {
                'search_module': colors['gold_dark'],
                'vector_search': colors['red_dark'],
                'web_search': colors['black'],
                'query_rewrite': colors['gold_medium'],
                'document_review': colors['red_medium']
            },
            'metrics': {
                'accuracy': colors['red_dark'],
                'time': colors['gold_dark'],
                'cost': colors['black']
            }
        }
    else:
        return get_color_scheme('figure_2')

def get_background_colors():
    """Get background color options"""
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

def get_sequential_colors(n_colors=5, theme='manchester_united'):
    """Get sequential colors for gradients"""
    if theme == 'vibrant':
        colors = ['#e53935', '#ff6b6b', '#ff8a80', '#ffcdd2', '#ffebee']
    elif theme == 'manchester_united':
        colors = ['#ffe4e9', '#ffc5ce', '#ff6b6b', '#d32f2f', '#0d0d0d']
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    return colors[:n_colors]

def get_categorical_colors(n_colors=6, theme='manchester_united'):
    """Get categorical colors for different groups"""
    if theme == 'vibrant':
        colors = ['#e53935', '#1e88e5', '#43a047', '#ff9800', '#8e24aa', '#00897b']
    elif theme == 'manchester_united':
        colors = ['#0d0d0d', '#ffc5ce', '#ffe697', '#ffe4e9', '#fff7f7', '#d32f2f']
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
    return colors[:n_colors]

def get_marker_styles():
    """Get marker styles"""
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
    """Get line styles"""
    return {
        1: ':',
        2: '--',
        3: '-',
        'solid': '-',
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.'
    }

def apply_standard_plot_formatting(ax, title_label=None, grid_alpha=0.2, grid_linewidth=0.5, background_color='white'):
    """Apply standard formatting to a plot axis"""
    if title_label:
        ax.set_title(title_label, fontsize=14, fontweight='bold', loc='left', pad=10)
    
    ax.set_facecolor(background_color)
    ax.grid(True, alpha=grid_alpha, linewidth=grid_linewidth)
    
    for label in [ax.get_xlabel(), ax.get_ylabel()]:
        if label:
            ax.set_xlabel(ax.get_xlabel(), fontweight='bold') if label == ax.get_xlabel() else None
            ax.set_ylabel(ax.get_ylabel(), fontweight='bold') if label == ax.get_ylabel() else None

def get_standard_figure_size():
    """Get standard figure size"""
    return (12, 8)

def get_standard_gridspec_params():
    """Get standard gridspec parameters"""
    return {
        'height_ratios': [1, 1],
        'width_ratios': [1.2, 1, 1],
        'hspace': 0.35,
        'wspace': 0.4
    }

def print_color_palette():
    """Print available colors"""
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