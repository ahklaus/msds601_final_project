import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

#Data Generation Functions

def generate_simpsons_data(n_groups=2, group_sizes=None, betas=None, sigma=1):
    if group_sizes is None:
        group_sizes = [100] * n_groups
    if betas is None:
        betas = [1] * n_groups
    
    X_all, Y_all, G_all = [], [], []
    
    for i in range(n_groups):
        n = group_sizes[i]
        beta = betas[i]
        X = np.random.rand(n) * 10
        noise = np.random.randn(n) * sigma
        Y = beta * X + noise
        X_all.append(X)
        Y_all.append(Y)
        G_all.append([f'Group {i+1}'] * n)
    
    df = pd.DataFrame({
        'X': np.concatenate(X_all),
        'Y': np.concatenate(Y_all),
        'Group': np.concatenate(G_all)
    })
    return df

def fit_regressions(df):
    group_lines = {}
    for group in df['Group'].unique():
        X_g = df[df['Group']==group]['X'].values.reshape(-1,1)
        Y_g = df[df['Group']==group]['Y'].values
        model = LinearRegression().fit(X_g, Y_g)
        group_lines[group] = (model.coef_[0], model.intercept_)
    
    X_all = df['X'].values.reshape(-1,1)
    Y_all = df['Y'].values
    agg_model = LinearRegression().fit(X_all, Y_all)
    agg_line = (agg_model.coef_[0], agg_model.intercept_)
    
    return group_lines, agg_line

def plot_simpsons(df, group_lines, agg_line):
    fig = go.Figure()
    
    for group in df['Group'].unique():
        df_g = df[df['Group']==group]
        fig.add_trace(go.Scatter(
            x=df_g['X'], y=df_g['Y'], mode='markers', name=group
        ))
        beta, intercept = group_lines[group]
        x_vals = np.array([df_g['X'].min(), df_g['X'].max()])
        y_vals = beta * x_vals + intercept
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name=f'{group} trend'
        ))
    
    beta, intercept = agg_line
    x_vals = np.array([df['X'].min(), df['X'].max()])
    y_vals = beta * x_vals + intercept
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines', name='Aggregate trend',
        line=dict(color='black', width=3)
    ))
    
    fig.update_layout(title="Simpson's Paradox Simulator",
                      xaxis_title='X', yaxis_title='Y')
    return fig


#Dash App

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Simpson's Paradox Simulation"),

    # Blog/Markdown section
    html.Div([
        html.H2("Understanding Simpson’s Paradox and Causal Inference"),
        dcc.Markdown("""
**Simpson’s Paradox** happens when a trend appears in different groups of data
but reverses or disappears when those groups are combined.

In this simulation:
- Each **group** has its own linear relationship between X and Y.
- The **aggregate regression** might show the *opposite* trend.
- This happens because the groups are **confounded** with X.

---

### What This Means for Causal Inference
Causal inference asks: *“Does X cause Y?”*

If we don’t account for confounders (like group membership),
we can mistakenly think X and Y are related — or miss the true relationship.
That’s why separating data by relevant subgroups is critical before making
causal claims.

**Play with the sliders**:
- Increasing the number of groups adds more confounders.
- Increasing noise adds random variation (unobserved factors).
- Watch how these affect the aggregate slope and each subgroup’s slope.
        """)
    ], style={
        'marginTop': 50,
        'maxWidth': '800px',
        'backgroundColor': '#f9f9f9',
        'padding': '40px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'
    }),

    # Graph and sliders section
    html.Div([
        dcc.Graph(id='scatter-plot'),

        html.Div([
            html.Label("Number of Groups"),
            dcc.Slider(id='num-groups', min=2, max=5, step=1, value=2),
            
            html.Label("Noise Level (sigma)"),
            dcc.Slider(id='noise', min=0, max=5, step=0.1, value=1),

            html.Label("Effect size per group (comma separated)"),
            dcc.Input(id='beta-input', value='1,-1', type='text', style={'width':'100%'}),
        ], style={'marginTop': 20})
    ]),

    html.Hr(),
])

#Call Back

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('num-groups', 'value'),
    Input('noise', 'value'),
    Input('beta-input', 'value')
)
def update_simpsons(n_groups, sigma, beta_text):
    try:
        betas = [float(x.strip()) for x in beta_text.split(',')]
    except:
        betas = [1]*n_groups
    
    if len(betas) != n_groups:
        betas = [1]*n_groups
    
    df = generate_simpsons_data(n_groups=n_groups, betas=betas, sigma=sigma)
    group_lines, agg_line = fit_regressions(df)
    fig = plot_simpsons(df, group_lines, agg_line)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
