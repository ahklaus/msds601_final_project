import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def compute_linear_fit(x_values, y_values):
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    x_mean = x.mean()
    y_mean = y.mean()
    x_var = np.var(x)
    if x_var == 0:
        return 0.0, float(y_mean)
    slope = float(np.cov(x, y, bias=True)[0, 1] / x_var)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def generate_simpsons_data(
    n_groups=2,
    points_per_group=150,
    betas=None,
    sigma=1.0,
    confound_strength=4.0,
    seed=None,
):
    if betas is None:
        betas = [1.0] * n_groups
    if len(betas) != n_groups:
        betas = [1.0] * n_groups

    x_width = 5.0
    separation = max(0.0, float(confound_strength))
    intercept_gap = float(confound_strength) * 2.0

    rng = np.random.default_rng(seed)

    all_x, all_y, all_groups = [], [], []
    for group_index in range(n_groups):
        n = int(points_per_group)
        beta = float(betas[group_index])

        x_start = group_index * separation
        x_vals = rng.random(n) * x_width + x_start
        intercept = (n_groups - 1 - group_index) * intercept_gap
        noise = rng.normal(0.0, float(sigma), size=n)
        y_vals = beta * x_vals + intercept + noise

        all_x.append(x_vals)
        all_y.append(y_vals)
        all_groups.append(np.array([f"Group {group_index + 1}"] * n))

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    slope_all, _ = compute_linear_fit(x, y)
    group_slopes = []
    for group_name in np.unique(groups):
        m = groups == group_name
        s, _ = compute_linear_fit(x[m], y[m])
        group_slopes.append(s)

    if len(group_slopes) > 0:
        within_sign = np.sign(np.median(group_slopes))
        agg_sign = np.sign(slope_all)
        tries = 0
        while within_sign != 0 and agg_sign == within_sign and tries < 3:
            intercept_gap *= 1.5
            y_list = []
            for group_index in range(n_groups):
                m = groups == f"Group {group_index + 1}"
                intercept = (n_groups - 1 - group_index) * intercept_gap
                y_list.append(
                    betas[group_index] * x[m]
                    + intercept
                    + rng.normal(0.0, float(sigma), size=m.sum())
                )
            y = np.concatenate(y_list)
            slope_all, _ = compute_linear_fit(x, y)
            agg_sign = np.sign(slope_all)
            tries += 1

    return x, y, groups


def build_simpsons_figure(x, y, groups):
    fig = go.Figure()

    unique_groups = np.unique(groups)
    palette = px.colors.qualitative.Safe
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(unique_groups)}

    for group_name in unique_groups:
        mask = groups == group_name
        x_g = x[mask]
        y_g = y[mask]
        color = color_map[str(group_name)]
        fig.add_trace(
            go.Scatter(
                x=x_g,
                y=y_g,
                mode="markers",
                name=str(group_name),
                marker=dict(size=7, opacity=0.75, color=color),
                hovertemplate="Group=%{legendgroup}<br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
                legendgroup=str(group_name),
            )
        )
        slope_g, intercept_g = compute_linear_fit(x_g, y_g)
        x_line = np.array([x_g.min(), x_g.max()])
        y_line = slope_g * x_line + intercept_g
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{group_name} trend",
                line=dict(color=color, width=3),
                hovertemplate=f"{group_name} trend<extra></extra>",
                legendgroup=str(group_name),
                showlegend=False,
            )
        )

    slope_all, intercept_all = compute_linear_fit(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope_all * x_line + intercept_all
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Aggregate trend",
            line=dict(color="#111111", width=4, dash="solid"),
            hovertemplate="Aggregate trend<extra></extra>",
        )
    )

    fig.update_layout(
        title=("Simpson's Paradox: Group vs Aggregate Trends"),
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, system-ui, -apple-system", size=15),
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
    )
    return fig


def compute_slopes(x, y, groups):
    unique_groups = np.unique(groups)
    group_to_slope = {}
    for group_name in unique_groups:
        m = groups == group_name
        s, _ = compute_linear_fit(x[m], y[m])
        group_to_slope[str(group_name)] = float(s)
    agg_slope, _ = compute_linear_fit(x, y)
    return group_to_slope, float(agg_slope)


