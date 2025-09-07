"""Component Breakdown Waterfall Chart - Simplified"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _parse_agent_cfg(exp_name: str):
    parts = exp_name.split('_')
    # Expect patterns like: "3_agents_1_round_with_search"
    n_agents = int(parts[0])
    n_rounds = int(parts[2])
    has_search = 'with_search' in exp_name
    return n_agents, n_rounds, has_search


def _mean_of(df, mask, col):
    s = df.loc[mask, col]
    return float(s.mean()) if len(s) else np.nan


def plot_component_breakdown_waterfall(ax, df, role_play_df, orchestration_df, colors, panel_label='D'):
    """Plot a simplified waterfall chart showing cumulative accuracy with time/cost overlays.

    - Bars: cumulative accuracy starting from baseline, then increments per component.
    - Lines (right axis): cumulative time and cost (cost scaled to per-100 for readability).
    """

    # Parse agent configuration columns once
    tmp = df['exp_name'].apply(_parse_agent_cfg).apply(pd.Series)
    tmp.columns = ['n_agents', 'n_rounds', 'has_search']
    df = pd.concat([df.copy(), tmp], axis=1)

    # Components in order
    components = [
        'Baseline',
        'Multi-Agent',
        'Evidence\nRetrieval',
        'Iterative\nReasoning',
        'Role Play',
        'Discussion\nOrchestration',
    ]

    # Baseline (1 agent, 1 round, no search)
    m_base_acc = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'accuracy')
    m_base_time = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'avg_time')
    m_base_cost = _mean_of(df, (df.n_agents == 1) & (df.n_rounds == 1) & (~df.has_search), 'avg_cost')

    # Multi-Agent: 3 agents, 1 round, no search
    m_ma_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'accuracy')
    m_ma_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'avg_time')
    m_ma_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (~df.has_search), 'avg_cost')

    # Evidence Retrieval: add search (3 agents, 1 round, with search)
    m_er_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'accuracy')
    m_er_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'avg_time')
    m_er_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 1) & (df.has_search), 'avg_cost')

    # Iterative Reasoning: add rounds (3 agents, 3 rounds, with search)
    m_ir_acc = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'accuracy')
    m_ir_time = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'avg_time')
    m_ir_cost = _mean_of(df, (df.n_agents == 3) & (df.n_rounds == 3) & (df.has_search), 'avg_cost')

    # Role Play: enable vs disable (use delta as increment)
    m_rp_enable_acc = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'accuracy')
    m_rp_disable_acc = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'accuracy')
    m_rp_enable_time = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'avg_time')
    m_rp_disable_time = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'avg_time')
    m_rp_enable_cost = _mean_of(role_play_df, role_play_df.exp_name == 'enable_role_play', 'avg_cost')
    m_rp_disable_cost = _mean_of(role_play_df, role_play_df.exp_name == 'disable_role_play', 'avg_cost')

    # Orchestration: best vs worst (use delta as increment)
    m_orch_best_acc = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'accuracy')
    m_orch_worst_acc = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'accuracy')
    m_orch_best_time = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'avg_time')
    m_orch_worst_time = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'avg_time')
    m_orch_best_cost = _mean_of(orchestration_df, orchestration_df.exp_name == 'group_chat_with_orchestrator', 'avg_cost')
    m_orch_worst_cost = _mean_of(orchestration_df, orchestration_df.exp_name == 'independent', 'avg_cost')

    # Build accuracy levels for the first 4 (cumulative path) and independent top values for RP/Orch
    acc_levels = [m_base_acc, m_ma_acc, m_er_acc, m_ir_acc, np.nan, np.nan]
    time_levels = [m_base_time, m_ma_time, m_er_time, m_ir_time, np.nan, np.nan]
    cost_levels = [m_base_cost, m_ma_cost, m_er_cost, m_ir_cost, np.nan, np.nan]

    # Deltas and bottoms for role play and orchestration (independent, not cumulative)
    rp_delta_acc = m_rp_enable_acc - m_rp_disable_acc
    rp_delta_time = m_rp_enable_time - m_rp_disable_time
    rp_delta_cost = m_rp_enable_cost - m_rp_disable_cost
    orch_delta_acc = m_orch_best_acc - m_orch_worst_acc
    orch_delta_time = m_orch_best_time - m_orch_worst_time
    orch_delta_cost = m_orch_best_cost - m_orch_worst_cost

    # For display, use top values for RP/Orch on lines
    acc_levels[4] = m_rp_enable_acc
    time_levels[4] = m_rp_enable_time
    cost_levels[4] = m_rp_enable_cost
    acc_levels[5] = m_orch_best_acc
    time_levels[5] = m_orch_best_time
    cost_levels[5] = m_orch_best_cost

    x = np.arange(len(components))
    comp_colors = colors['component_breakdown']
    bar_palette = [
        comp_colors['Baseline'],
        comp_colors['Multi-Agent'],
        comp_colors['Evidence_Retrieval'],
        comp_colors['Iterative_Reasoning'],
        comp_colors['Role_Play'],
        comp_colors['Discussion_Orchestration'],
    ]

    # Draw accuracy waterfall
    # First 4 bars are cumulative steps based on absolute levels; RP/Orch are independent deltas from their own bottoms
    running = 0.0
    for i, color in enumerate(bar_palette):
        if i == 0:
            bottom = 0.0
            height = acc_levels[0]
        elif i in (1, 2, 3):
            bottom = acc_levels[i - 1]
            height = acc_levels[i] - acc_levels[i - 1]
        elif i == 4:
            bottom = m_rp_disable_acc
            height = rp_delta_acc
        else:  # i == 5
            bottom = m_orch_worst_acc
            height = orch_delta_acc
        ax.bar(x[i], height, bottom=bottom, color=color, width=0.65, edgecolor='black', linewidth=1.0, alpha=0.9)

    # Left axis labels and styling (match 0.B)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=15, rotation=10)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, alpha=0.4, axis='y', linewidth=0.8, linestyle=':')
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')

    # Right axis: time and cost (per 100)
    ax2 = ax.twinx()
    time_color = colors['methods'].get('Few-shot', '#444444')
    cost_color = colors['methods'].get('MedRAG', '#888888')
    time_line = ax2.plot(x, time_levels, color=time_color, linestyle='-', marker='D',
                         markersize=10, linewidth=4, alpha=0.9,
                         label='Execution Time (s)', markeredgecolor='white', markeredgewidth=1)
    cost_line = ax2.plot(x, [c * 100 for c in cost_levels], color=cost_color, linestyle='--', marker='^',
                         markersize=10, linewidth=4, alpha=0.9,
                         label='Cost (per 100 samples)', markeredgecolor='white', markeredgewidth=1)
    ax2.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=18)
    ax2.tick_params(axis='both', labelsize=15)

    # Loosen right-axis limits so labels always fit
    y2_values = [v for v in list(time_levels) + [vv * 100.0 for vv in cost_levels] if np.isfinite(v)]
    if y2_values:
        y2_min, y2_max = min(y2_values), max(y2_values)
        y2_pad = max(5.0, 0.15 * (y2_max - y2_min))
        ax2.set_ylim(y2_min - y2_pad, y2_max + y2_pad)

    # Legend styling: move to upper-left, no redundant title
    handles, labels = ax2.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              fontsize=13, ncol=1, frameon=True, fancybox=True, shadow=True,
              framealpha=0.95, facecolor='white', edgecolor='black')

    # Limits with padding
    ymin, ymax = np.nanmin(acc_levels), np.nanmax(acc_levels)
    pad = (ymax - ymin) * 0.15 if np.isfinite(ymin) and np.isfinite(ymax) else 5
    ax.set_ylim(max(0, ymin - pad), ymax + pad)
    ax.set_xlim(-0.5, len(components) - 0.5)

    # Annotations: bars (accuracy deltas) and lines (time, cost) with overlap-avoidance
    # Precompute right-axis range for heuristic spacing
    y2_vals = [v for v in list(time_levels) + [vv * 100.0 for vv in cost_levels] if np.isfinite(v)]
    y2_min, y2_max = (min(y2_vals), max(y2_vals)) if y2_vals else (0.0, 1.0)
    y2_thr = 0.06 * (y2_max - y2_min)  # threshold for “close” labels

    for i in range(len(components)):
        # Bar annotation (left axis) — put slightly above bar top
        if i == 0:
            top_y = acc_levels[0]
            label = f"{top_y:.1f}%"
        elif i in (1, 2, 3):
            delta = acc_levels[i] - acc_levels[i - 1]
            top_y = acc_levels[i]
            label = f"+{delta:.1f}%" if np.isfinite(delta) else ""
        elif i == 4:
            top_y = m_rp_disable_acc + max(0, rp_delta_acc)
            label = f"+{rp_delta_acc:.1f}%"
        else:  # 5
            top_y = m_orch_worst_acc + max(0, orch_delta_acc)
            label = f"+{orch_delta_acc:.1f}%"

        if label:
            ax.annotate(label, xy=(x[i], top_y), xycoords='data',
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', clip_on=True)

        # Line annotations (right axis) — alternate left/right and separate time vs cost
        t = time_levels[i]
        c = cost_levels[i] * 100.0
        # Default offsets: time up-left, cost down-right (alternate per index)
        t_dx = -12 if i % 2 == 0 else 12
        t_dy = 10
        c_dx = 12 if i % 2 == 0 else -12
        c_dy = -12
        # If time and cost are very close, push them further apart vertically
        if np.isfinite(t) and np.isfinite(c) and abs(t - c) < y2_thr:
            t_dy = 12
            c_dy = -18

        if np.isfinite(t):
            ax2.annotate(f"{t:.0f}s", xy=(x[i], t), xycoords='data',
                         textcoords='offset points', xytext=(t_dx, t_dy),
                         ha='right' if t_dx < 0 else 'left', va='bottom', fontsize=10,
                         color=time_color,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=time_color, alpha=0.85),
                         clip_on=True)
        if np.isfinite(c):
            ax2.annotate(f"${c:.1f}", xy=(x[i], c), xycoords='data',
                         textcoords='offset points', xytext=(c_dx, c_dy),
                         ha='left' if c_dx > 0 else 'right', va='top', fontsize=10,
                         color=cost_color,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=cost_color, alpha=0.85),
                         clip_on=True)

    return ax


def main():
    # Minimal local demo when running this file directly
    from plot_utils import apply_medagents_style, get_figure_0_colors
    apply_medagents_style()
    colors = get_figure_0_colors()

    df = pd.DataFrame({
        'exp_name': [
            '1_agent_1_round_no_search',
            '3_agents_1_round_no_search',
            '3_agents_1_round_with_search',
            '3_agents_3_rounds_with_search',
        ],
        'accuracy': [20, 25, 30, 32],
        'avg_time': [15, 35, 65, 105],
        'avg_cost': [0.1, 0.2, 0.35, 0.55],
    })

    role_play_df = pd.DataFrame({
        'exp_name': ['enable_role_play', 'disable_role_play'],
        'accuracy': [35, 30],
        'avg_time': [60, 45],
        'avg_cost': [0.46, 0.44],
    })

    orchestration_df = pd.DataFrame({
        'exp_name': ['group_chat_with_orchestrator', 'independent'],
        'accuracy': [35, 28],
        'avg_time': [70, 45],
        'avg_cost': [0.50, 0.13],
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_component_breakdown_waterfall(ax, df, role_play_df, orchestration_df, colors, panel_label='D')
    plt.tight_layout()
    plt.savefig('component_breakdown_waterfall_example.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
