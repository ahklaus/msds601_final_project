import numpy as np
import plotly.graph_objects as go


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
):
    if betas is None:
        betas = [1.0] * n_groups
    if len(betas) != n_groups:
        betas = [1.0] * n_groups

    x_width = 5.0
    separation = max(0.0, float(confound_strength))
    intercept_gap = float(confound_strength) * 2.0

    all_x, all_y, all_groups = [], [], []
    for group_index in range(n_groups):
        n = int(points_per_group)
        beta = float(betas[group_index])

        x_start = group_index * separation
        x_vals = np.random.rand(n) * x_width + x_start
        intercept = (n_groups - 1 - group_index) * intercept_gap
        noise = np.random.randn(n) * float(sigma)
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
                    + np.random.randn(m.sum()) * float(sigma)
                )
            y = np.concatenate(y_list)
            slope_all, _ = compute_linear_fit(x, y)
            agg_sign = np.sign(slope_all)
            tries += 1

    return x, y, groups


def build_simpsons_figure(x, y, groups):
    fig = go.Figure()

    unique_groups = np.unique(groups)
    for group_name in unique_groups:
        mask = groups == group_name
        x_g = x[mask]
        y_g = y[mask]
        fig.add_trace(go.Scatter(x=x_g, y=y_g, mode="markers", name=str(group_name)))
        slope_g, intercept_g = compute_linear_fit(x_g, y_g)
        x_line = np.array([x_g.min(), x_g.max()])
        y_line = slope_g * x_line + intercept_g
        fig.add_trace(
            go.Scatter(x=x_line, y=y_line, mode="lines", name=f"{group_name} trend")
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
            line=dict(color="black", width=3),
        )
    )

    fig.update_layout(
        title=("Simpson's Paradox: Group vs Aggregate Trends"),
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )
    return fig


