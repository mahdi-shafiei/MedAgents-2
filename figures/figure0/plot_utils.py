"""Common plotting utilities and color schemes"""
import matplotlib.pyplot as plt

def get_manchester_colors():
    """Get Manchester United Official color scheme"""
    return {
        "black": "#000000",
        "jet": "#444444", 
        "gray": "#888888",
        "french_gray": "#BCB9BE",
        "snow": "#FFF0F7",
        "penn_red": "#BB3234",
        "barn_red": "#A40000",
        "engineering_orange": "#D26367",
        "pink": "#FFC5CE",
        "jasmine": "#FFE697",
        "sunset": "#F9C578",
        "sandy_brown": "#F2A358",
        "sandy_brown_light": "#F8D1AB",
        "persimmon": "#F47B3C"
    }

def get_figure_0_colors():
    """Get Figure 0 specific color mapping"""
    manchester_colors = get_manchester_colors()
    return {
        'methods': {
            'MedAgents-2': manchester_colors['jasmine'],
            'AFlow': manchester_colors['sandy_brown_light'],
            'SPO': manchester_colors['sunset'],
            'MultiPersona': manchester_colors['sandy_brown'],
            'Self-refine': manchester_colors['persimmon'],
            'CoT': manchester_colors['barn_red'],
            'CoT-SC': manchester_colors['penn_red'],
            'MedAgents': manchester_colors['pink'],
            'MDAgents': manchester_colors['snow'],
            'MedPrompt': manchester_colors['french_gray'],
            'MedRAG': manchester_colors['gray'],
            'Few-shot': manchester_colors['jet'],
            'Zero-shot': manchester_colors['black']
        },
        'metrics': {
            'accuracy': manchester_colors['penn_red'],
            'time': manchester_colors['jasmine'],
            'cost': manchester_colors['black']
        },
        'annotations': {
            'text_box_face': manchester_colors['snow'],
            'text_box_edge': manchester_colors['gray'],
            'text': manchester_colors['black']
        },
        'component_breakdown': {
            'Baseline': manchester_colors['gray'],
            'Multi-Agent': manchester_colors['jet'],
            'Evidence_Retrieval': manchester_colors['sandy_brown'],
            'Iterative_Reasoning': manchester_colors['sunset'],
            'Role_Play': manchester_colors['penn_red'],
            'Discussion_Orchestration': manchester_colors['jasmine'],
            'Final': manchester_colors['barn_red']
        }
    }

def get_figure_1_colors():
    """Get Figure 1 specific color mapping"""
    manchester_colors = get_manchester_colors()
    return {
        'no_search': manchester_colors['jet'],
        'with_search': manchester_colors['penn_red'],
        'rounds': {1: manchester_colors['penn_red'], 2: manchester_colors['sandy_brown'], 3: manchester_colors['jasmine']},
        'architecture': {
            'agent': manchester_colors['black'],
            'search': manchester_colors['sandy_brown'],
            'reasoning': manchester_colors['sunset'],
            'knowledge': manchester_colors['penn_red']
        },
        'metrics': {
            'accuracy': manchester_colors['penn_red'],
            'time': manchester_colors['sandy_brown'],
            'cost': manchester_colors['black']
        },
        'role_play': {
            'enable_role_play': manchester_colors['penn_red'],
            'disable_role_play': manchester_colors['gray']
        },
        'orchestration': {
            'group_chat_with_orchestrator': manchester_colors['penn_red'],
            'group_chat_voting_only': manchester_colors['sunset'],
            'independent': manchester_colors['gray'],
            'one_on_one_sync': manchester_colors['jasmine']
        }
    }

def get_figure_2_colors():
    """Get Figure 2 specific color mapping"""
    manchester_colors = get_manchester_colors()
    return {
        'modality': {
            'both': manchester_colors['penn_red'],
            'vector_only': manchester_colors['jasmine'],
            'web_only': manchester_colors['black'],
            'none': manchester_colors['gray']
        },
        'features': {
            'baseline': manchester_colors['penn_red'],
            'no_document_review': manchester_colors['sunset'],
            'no_query_rewrite': manchester_colors['jet'],
            'no_rewrite_no_review': manchester_colors['gray']
        },
        'history': {
            'individual': manchester_colors['penn_red'],
            'shared': manchester_colors['sandy_brown']
        },
        'depth': {
            'more_docs': manchester_colors['penn_red']
        },
        'metrics': {
            'accuracy': manchester_colors['penn_red'],
            'time': manchester_colors['jasmine'],
            'cost': manchester_colors['black']
        },
        'annotations': {
            'text_box_face': manchester_colors['snow'],
            'text_box_edge': manchester_colors['gray'],
            'text': manchester_colors['black']
        }
    }

def apply_medagents_style():
    """Apply MedAgents plotting style"""
    try:
        plt.style.use('/home/ubuntu/MedAgents-2/figures/medagents.mplstyle')
    except:
        # Fallback to basic style if mplstyle not available
        plt.rcParams.update({
            'font.size': 8.5,
            'font.family': 'sans-serif',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.15,
            'legend.frameon': False,
            'lines.linewidth': 1.8,
            'figure.dpi': 300
        })