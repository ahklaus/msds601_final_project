import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simpsons import (
    generate_simpsons_data,
    build_simpsons_figure,
    compute_slopes,
)
from dash.dependencies import ALL

"""Interactive blog + Simpson's Paradox simulation."""

# Initialize app (Material-like theme)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])
app.title = "Causal Inference in Regression — Interactive Blog"
server = app.server

# ---------- BLOG CONTENT SECTION ----------
# You can paste your blog paragraphs here as Markdown.
# Use ###, ####, and **bold/italic** syntax for structure.

md_header = dcc.Markdown(r"""
# Causal Inference in Regression
*MSDS601 Final Project by Amelia Klaus, Rodrigo Cuadra, & Rory Mackintosh *  

---
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_two_faces = dcc.Markdown(r"""
## The Two Faces of Regression: Association and Causation  
Ask anyone with a high school diploma what they remember and you’ll usually get one of two answers: “the mitochondria is the powerhouse of the cell” or “correlation is not causation.” While the biology fact still stands, our focus today is on the second truth. In principle, the phrase is quite self-explanatory: correlation is merely an association of two pieces of data whereas causation occurs when one event directly contributes to the occurrence of the other. 

![Correlation vs Causation](/assets/image1.png)

There are countless instances of correlated events not having a causal relationship, such as the increase in associate degrees awarded in science technologies and Google searches for “avocado toast.” Does this mean that as students developed their engineering skills their entire career was leading to the breakthrough of engineering the perfect avocado toast? Not quite. More of these examples can be found on [Tyler Vigen’s project on *Spurious Correlations*](https://www.tylervigen.com/spurious-correlations) [^1].

 
Though in reverse, we cannot have causation without correlation. This leads us to question the other requirements for causal relationships, at what point can we say the perfection of avocado toast was caused by the increase in associate degrees awarded in science technologies?

Before we get there, it’s important to look at the inferences of regression, specifically, the difference between prediction and causal inference. Prediction involves the comparison of outcomes between different units. An example of this would be comparing the test scores of two students: one who studied for the test and one who did not. Causal inference takes the same unit and examines multiple outcomes. How would student A’s test score differ if they studied versus if they didn’t? This type of inference addresses the issue of “correlation is not causation” because it fixates on cause-and-effect relationships to draw conclusions beyond statistical association. 

Standard regression is a tool built for prediction, not causation. So to be able to draw causal conclusions from regression models, we must focus on strict assumptions and frameworks. Going forward, we will tackle the fundamental problem with causal inference, a framework to define and identify causal effects, and a real-world example using Simpson’s Paradox.
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_fundamental = dcc.Markdown(r"""
### The Fundamental Problem
  
Now where exactly is the problem with causal inference? To visualize this, I like to think about the butterfly effect. Imagine you’re a student who realized too late that there’s an exam tomorrow (perhaps for a particular regression class). Your mind is split between two choices:

- Choice 1: Do not study. Get a solid night’s sleep and hope that a rested mind will carry you through it (or you’re just drawing bunnies the entire time).
- Choice 2: Study. Sacrifice sleep, cram all night, and hope the extra studying pays off.

In your head, you can play out both timelines. Maybe in choice 1, you bomb the exam and flunk out of your program. Alternatively, in choice 2, you ace the exam and the class, impress your professor, and land your dream job on behalf of your academic merit.

But in reality, you only get to live one of those outcomes. In this case, you either sleep or you don't, which determines the test score you receive. The other path is forever unknowable. Introducing the fundamental problem of causal inference: for any individual, we can only observe the outcome of the decision they actually made and not the alternative reality. Ideally, we want to know how outcomes (test scores here) would change due to different treatments (studying). Regression only provides access to the single outcome of the choice that was made, but the counterfactuals, or the roads not taken, are always missing.  
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_slr_mlr = dcc.Markdown(r"""
### SLR, MLR, and Multicollinearity
In the context of simple linear regression, take the example above of predicting exam scores, $Y$, from study hours, $X_1$. To build a simple regression model we hypothesize the model to be of the form

$$Y_i = \beta_0 + \beta_1 X_{1i} + \varepsilon_i$$

Though, since we will never truly know the values of $\beta_0$ and $\beta_1$ we create unbiased estimates:

$$\hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_{1i}$$

Because of the assumptions we’ve made, our interpretation of the model is limited. We say that $\hat{\beta}_1$ estimates the average change in test score for every one-hour increase in study hours. But this is all a measure of association, not causation. The slope tells us what happens when X changes, but not why this occurs. This increase could be the result of something else, maybe students who study more attend class more or have more genuine interest in the subject matter. These hidden causes that affect both the predictor and the outcome are known as confounding variables.

From the perspective of causal inference, regression coefficients identify patterns in data, through direction and strength of relationships, but not the mechanisms that generate them. To move towards causal inference, we shift our focus to the changes in Y resulting from X, rather than the changes in X associated with Y.

However, this doesn’t mean regression coefficients are useless for causal work, we can never prove causation if we have not first identified association. It simply means we look beyond standard errors and t-tests, but rather the assumptions that went into collecting the data.

Now, returning to our model above, let’s add a confounder: a student’s interest in the subject, denoted by Z. This variable affects both study time and performance.

$$Y_i = \beta_0 + \beta_1(X_i) + \beta_2(Z_i) + \epsilon_i$$

If we look back at our proposed simpler model, the ordinary least squares estimate for $\beta_1$ becomes biased.

The simple regression estimate $\tilde{\beta}_1$ is biased because X (study hours) and the omitted variable Z (interest) are correlated.

$$\tilde{\beta}_1 = \beta_1 + \beta_2 \frac{\text{Cov}(X,Z)}{\text{Var}(X)}$$

This illustrates how omitted variables introduce bias: unless $\beta_2 = 0$ (interest has no effect on exam performance) or Cov(X,Z) = 0 (interest is independent of study hours) then our estimated slope for study hours ($\hat{\beta}_1$) is biased [^2].  

The solution here is multiple linear regression (MLR) where we can include all confounding variables.   

** The Case of Multiple Linear Regression (MLR) and Multicollinearity**

In multiple regression, the variance of the estimated coefficient $\hat{\beta}_1$ is determined by the structure of $X$. Specifically,  

$$Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}$$

(under the assumption of homoskedastic errors, where $Var(\varepsilon) = \sigma^2I$).

This tells us that the precision of our estimated coefficient is influenced by the independence of each predictor from the others. If two regressors X and Z are highly correlated, the matrix $X^TX$ becomes nearly singular, and the variance formula blows up. Intuitively, as two variables move together, the model has trouble distinguishing their individual effects, leading to unstable estimates and uncertainty.

Note that perfect collinearity is impossible to estimate as the model would be undefined, but severe collinearity makes standard errors large and interpretations unreliable.

This means that while including Z (interest) corrects bias in theory, it destabilizes the estimate of the causal effect by inflating uncertainty because the variance inflation factor (VIF) increases with multicollinearity.

** Multicollinearity**

This is where prediction models differ from causal models. With causal models, our overarching goal is to reduce bias so we can make that ultimate causal inference. To achieve this, we must ensure we have an unbiased causal estimator, which means all the relevant confounders are included in the model.

Does this mean that causal models favor multicollinear terms? Not necessarily. 

In MLR for prediction, multicollinearity is a numerical issue. It makes coefficients harder to interpret, though the predictions remain unbiased, just messier.  

Now looking at MLR for causation, we want to know if the variation in $X$ is independent (enough) of confounders. If $X$ and our confounder $Z$ are nearly collinear, then after we control $Z$ we are left with little independent variation in $X$. Mathematically,

$$\hat{\beta}_1 = \frac{Cov(Y,X|Z)}{Var(X|Z)}$$

If $Var(X|Z)$ is small (because they move together), we don't have as much information to estimate the causal effect. Here, multicollinearity results in the loss of the ability to identify causality. 
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_rubin = dcc.Markdown(r"""
### Rubin's Causal Model

Causal inference boils down to a missing data problem. How do we estimate what we can't see? This question is foundational to the Rubin Causal Model (RCM), which defines causal effects as a comparison of what would have happened to the same units under different treatments. Let's change our example and instead of studying, we've discovered a drug that condenses an entire module's worth of content into one digestible pill. We're still interested in test scores, but we want to see if our drug works.

Let's consider the simple case of a binary treatment, where $T_i$ represents the treatment status of the *i*th unit. $Y_i$ has two potential outcomes:

Let $Y(1)$ = potential outcome **under treatment** (uses drug and does not study).

And $Y(0)$ = potential outcome **under control** (does not use drug and does not study).

Before the treatment is applied, both $Y(1)$ and $Y(0)$ exist as potentials for every unit. After intervention, only one of these is observed [^3].

This gives us

$$Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)$$

The treatment effect for the *i*th unit can be defined as $Y_i(1) - Y_i(0)$, which measures the gain in the outcome variable (test scores) under assignment to the treatment. To statistically overcome the fundamental problem of causal inference, we want to generalize to estimate the average treatment effect rather than an individual one.

$$\theta = E\{Y_i(1)\} - E\{Y_i(0)\}$$

Under certain assumptions, we can express these expectations using observed data:

$$E\{Y_i(1)\} = E[\,E(Y_i \mid T_i = 1, X_i)\,], \quad E\{Y_i(0)\} = E[\,E(Y_i \mid T_i = 0, X_i)\,]$$

The **independence assumption** ensures that

$$\theta = E[\,E(Y_i \mid T_i = 1, X_i) - E(Y_i \mid T_i = 0, X_i)\,]$$

provides an unbiased estimate for $\theta$. We still must address **strong ignorability** [^4].

As defined by the RCM, to make the assumption of strong ignorability, the following must hold for each *i*:

(i) **Unconfoundedness**

$$ (Y_i(1), Y_i(0)) \perp T_i \mid X_i $$

This is the conditional independence of treatment assignment given the pretreatment variables. In simple terms, the assignment to treatment ($T$) is independent of the outcomes after conditioning on the pre-treatment covariates. Within a subpopulation defined by the same covariate $(X_i = x)$, the assignment to treatment is as good as random, meaning it is fair to compare both treatment and control groups at that level of $X$.

When this condition holds, we are able to estimate the average treatment effect for a given subpopulation by comparing the average observed outcomes between treated and control units within that group. Here's what we've built mathematically:

$$\theta(x) = E\{Y_i(1) \mid T_i = 1, X_i = x\} - E\{Y_i(0) \mid T_i = 0, X_i = x\}$$

Once unconfoundedness is satisfied, we can average $\theta(x)$ over all possible $X$ values to get an unbiased estimator of $\theta$, the overall average treatment effect:

$$\theta = E_X[\theta(X)]$$

(ii) **Overlap**

$$0 < P(T_i = 1 \mid X_i = x) < 1$$

The **propensity score** is defined as $e(x) = P(T_i = 1 \mid X_i = x)$. For every set of pre-treatment characteristics $X_i = x$, there must be a non-zero probability that a unit with those characteristics receives the treatment *and* that they receive the control. If this assumption were violated, it would be impossible to estimate the potential outcome of the unobserved group.

---

The idea behind all of this is that **randomization implies ignorability**, giving us the power to say that assignment to treatment is independent of potential outcomes.

**Warning! Do NOT control post-treatment variables**

Though it was perfectly fine to take pre-treatment variables into account, variables measured before the treatment were not impacted by the treatment, but measuring a variable after treatment can introduce bias and invalidate the causal estimate, even if we properly randomized! This is because we are comparing groups that are fundamentally different. Comparing treatment and control groups on the basis of some outcome score is inherently biased because each individual likely had different potentials long before the experiment took place [^5].
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_simpson = dcc.Markdown(r"""
### Simpson's Paradox
To put together all that we’ve discussed so far, let’s look at a real-world example of how regression can be misleading through Simpson’s Paradox. Simpson’s Paradox occurs where an association, or trend, that appears in an entire population is reversed when looking at subpopulations. For instance, a simple linear regression might show that X increases Y, but when we split the data by another variable, Z, each subgroup shows the opposite trend [^6].

How can both of these associations be true? The answer lies in the confounding variables we discussed earlier. Since Z influences both X and Y, it distorts their relationship when combining data. By identifying Z, we’ve found a true causal predictor as the relationship between X and Y fades. Simpson’s Paradox is less of a paradox, and more of a design problem. Ignoring our confounders can lead to a completely different analysis of our data.
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, mathjax=True, link_target="_blank")

md_sim_header = dcc.Markdown(r"""
## Simpson's Paradox: An Interactive Simulation
Here’s where you transition to your interactive component.
Explain what the user will be able to explore below.

---
""", style={"fontSize": "16px", "lineHeight": "1.7em"}, link_target="_blank")


# ---------- INTERACTIVE DASHBOARD SECTION ----------
# Simulation helpers
from simpsons import compute_linear_fit  # only if needed elsewhere


controls = dbc.Card(
    [
        html.Div(
            [
                html.Div([
                    html.H4("Simpson's Paradox Simulator", className="card-title", style={"marginBottom": "2px"}),
                    html.Div("Explore how group-wise trends can reverse in aggregate.", style={"color": "#6c757d"}),
                ], className="hero"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button("Randomize", id="btn-randomize", color="primary", className="me-1"),
                        dbc.Button("Reset", id="btn-reset", outline=True, color="secondary", className="btn-ghost"),
                    ], md=8),
                    dbc.Col([
                        dbc.DropdownMenu(
                            label="Presets",
                            children=[
                                dbc.DropdownMenuItem("All positive", id="preset-pos"),
                                dbc.DropdownMenuItem("Mixed signs", id="preset-mix"),
                                dbc.DropdownMenuItem("All negative", id="preset-neg"),
                            ],
                            color="light",
                            className="float-end"
                        )
                    ], md=4),
                ], className="mb-2"),
            ]
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Groups", style={"fontWeight": 600}),
                dcc.Slider(id="num-groups", min=2, max=5, step=1, value=2, marks=None, tooltip={"placement": "bottom", "always_visible": False}),
            ], md=6),
            dbc.Col([
                html.Label("Points / group", style={"fontWeight": 600}),
                dcc.Slider(id="points-per-group", min=50, max=500, step=25, value=200, marks=None, tooltip={"placement": "bottom", "always_visible": False}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("Noise (sigma)", style={"fontWeight": 600}),
                dcc.Slider(id="noise", min=0, max=5, step=0.1, value=1.0, marks=None, tooltip={"placement": "bottom", "always_visible": False}),
            ], md=6),
            dbc.Col([
                html.Label("Confounding", style={"fontWeight": 600}),
                dcc.Slider(id="confound", min=0, max=10, step=0.5, value=4.0, marks=None, tooltip={"placement": "bottom", "always_visible": False}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("Betas", style={"fontWeight": 600}),
                html.Div(id="beta-inputs"),
            ], md=8),
            dbc.Col([
                html.Label("Seed", style={"fontWeight": 600}),
                dcc.Input(id="seed", value="", type="text", placeholder="Optional", style={"width": "100%"}),
            ], md=4),
        ]),

        html.Div(id="slope-summary", style={"marginTop": "12px", "color": "#495057"}),
    ],
    body=True,
)

graph_card = dbc.Card(
    [
        dcc.Graph(id="sim-graph", config={"displayModeBar": True, "displaylogo": False}, style={"height": "600px"}),
    ],
    body=True,
)

simpson_section = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(graph_card, md=9),
            ],
            align="center",
            className="g-2",
        )
    ],
    fluid=True,
    style={"maxWidth": "1200px", "padding": "0 12px"},
)

md_conclusion = dcc.Markdown(
    """

---

## Conclusion  
We began with the age-old warning that “correlation is not causation” where we unpacked this. Along the way, we saw regression is inherently predictive, not causal, as we have to dig deeper to find the “why.” This led us to Rubin’s Causal Model, where we consider two outcomes of each unit, and recognize that only one will ever be observed. This causes us to rely on the assumption of ignorability and independence so we can achieve the gold standard of randomization in real-world data. Overall, causality boils down to a design problem. The most critical part of causal analysis happens long before .fit() is used as the computer will crunch the numbers regardless, but it’s the job of the researcher to design the experiment. If the structure of data collection is not fit for causal inference, no amount of statistical manipulation will change this.

In prediction, we tend to select models through metrics such as Akaike’s Information Criterion (AIC), or adjusted R-squared. But these reward the models that fit the observed data the best, not the best practices of data generation. A model can appear statistically significant, but be causally meaningless if correlation is confused for causation. Meaning, maximizing prediction can sometimes hide the causal bias we’re trying to eliminate. True causal inference relies on theoretical reasoning, not just statistical fit. 

In short, the best predictive model isn’t always the best causal model. Regression helps us describe the world as it is, but causal inference helps us imagine the world that could’ve been. So we end here where we began: correlation may start the story, but only causation writes the plot.

---

*Thanks for reading!*
""",
    style={"fontSize": "16px", "lineHeight": "1.7em"}, link_target="_blank"
)

md_references = dcc.Markdown(
    """
## References

[^1]: Vigen, Tyler. "Spurious Correlations." Accessed October 7, 2025. https://www.tylervigen.com/spurious-correlations.

[^2]: Angrist, Joshua D. “The Long and Short of OVB.” Lecture notes, 2020. Mastering Metrics online resources. https://www.masteringmetrics.com/wp-content/uploads/2020/07/lny20n08MRU_R2.pdf

[^3]: Gelman, Andrew, Jennifer Hill, and Aki Vehtari. *Regression and Other Stories* (Cambridge: Cambridge University Press, 2020), [page number].

[^4]: Rubin, Donald B. "For Objective Causal Inference, Design Trumps Analysis." *Annals of Applied Statistics* 2, no. 3 (2008): 808-840. https://doi.org/10.1214/08-AOAS187.

[^5]: Pearl, Judea, Madelyn Glymour, and Nicholas P. Jewell. *Causal Inference in Statistics: A Primer*. Chichester, UK: Wiley, 2016.

[^6]: Selvitella, Alessandro. "The Ubiquity of the Simpson's Paradox." *Journal of Statistical Distributions and Applications* 4, no. 2 (2017). https://doi.org/10.1186/s40488-017-0056-5.
    """,
    style={"fontSize": "16px", "lineHeight": "1.7em"}, link_target="_blank"
)

# ---------- NEW INTERACTIVE: OVB / Multicollinearity ----------
ovb_controls = dbc.Card(
    [
        html.Div([
            html.H4("OVB & Multicollinearity Explorer", className="card-title", style={"marginBottom": "2px"}),
            html.Div("Compare SLR vs MLR when X and Z are correlated.", style={"color": "#6c757d"}),
        ], className="hero"),

        dbc.Row([
            dbc.Col([
                html.Label("Sample size", style={"fontWeight": 600}),
                dcc.Slider(id="ovb-n", min=100, max=5000, step=100, value=1000, tooltip={"placement": "bottom"}),
            ], md=6),
            dbc.Col([
                html.Label("Noise (sigma)", style={"fontWeight": 600}),
                dcc.Slider(id="ovb-sigma", min=0, max=3, step=0.05, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("β1 (effect of X)", style={"fontWeight": 600}),
                dcc.Slider(id="ovb-beta1", min=-3, max=3, step=0.1, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
            dbc.Col([
                html.Label("β2 (effect of Z)", style={"fontWeight": 600}),
                dcc.Slider(id="ovb-beta2", min=-3, max=3, step=0.1, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("Corr(X,Z) = ρ", style={"fontWeight": 600}),
                dcc.Slider(id="ovb-rho", min=-0.95, max=0.95, step=0.05, value=0.6, tooltip={"placement": "bottom"}),
            ], md=8),
            dbc.Col([
                html.Label("Seed", style={"fontWeight": 600}),
                dcc.Input(id="ovb-seed", type="text", value="", placeholder="Optional", style={"width": "100%"}),
            ], md=4),
        ]),

        html.Div([
            dbc.Button("Randomize", id="ovb-rand", color="primary", className="me-1"),
            dbc.Button("Reset", id="ovb-reset", outline=True, color="secondary", className="btn-ghost"),
        ], className="mt-2"),

        html.Div(id="ovb-summary", style={"marginTop": "12px", "color": "#495057"}),
    ],
    body=True,
)

ovb_graph_card = dbc.Card([
    dcc.Graph(id="ovb-graph", config={"displayModeBar": True, "displaylogo": False}, style={"height": "600px"}),
], body=True)

ovb_section = dbc.Container([
    dbc.Row([
        dbc.Col(ovb_controls, md=3),
        dbc.Col(ovb_graph_card, md=9),
    ], className="g-2", align="center")
], fluid=True, style={"maxWidth": "1200px", "padding": "0 12px"})


# ---------- NEW INTERACTIVE: Rubin Causal Model ----------
rcm_controls = dbc.Card(
    [
        html.Div([
            html.H4("Rubin Causal Model Explorer", className="card-title", style={"marginBottom": "2px"}),
            html.Div("See bias from confounded assignment and IPW correction.", style={"color": "#6c757d"}),
        ], className="hero"),

        dbc.Row([
            dbc.Col([
                html.Label("Sample size", style={"fontWeight": 600}),
                dcc.Slider(id="rcm-n", min=200, max=10000, step=200, value=2000, tooltip={"placement": "bottom"}),
            ], md=6),
            dbc.Col([
                html.Label("Noise (sigma)", style={"fontWeight": 600}),
                dcc.Slider(id="rcm-sigma", min=0, max=3, step=0.05, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("τ (true ATE)", style={"fontWeight": 600}),
                dcc.Slider(id="rcm-tau", min=-3, max=3, step=0.1, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
            dbc.Col([
                html.Label("α (effect of X)", style={"fontWeight": 600}),
                dcc.Slider(id="rcm-alpha", min=-3, max=3, step=0.1, value=1.0, tooltip={"placement": "bottom"}),
            ], md=6),
        ]),

        html.Br(),

        dbc.Row([
            dbc.Col([
                html.Label("Assignment bias k (in e(x)=sigmoid(k·X))", style={"fontWeight": 600}),
                dcc.Slider(id="rcm-k", min=-5, max=5, step=0.1, value=2.0, tooltip={"placement": "bottom"}),
            ], md=8),
            dbc.Col([
                html.Label("Seed", style={"fontWeight": 600}),
                dcc.Input(id="rcm-seed", type="text", value="", placeholder="Optional", style={"width": "100%"}),
            ], md=4),
        ]),

        html.Div([
            dbc.Button("Randomize", id="rcm-rand", color="primary", className="me-1"),
            dbc.Button("Reset", id="rcm-reset", outline=True, color="secondary", className="btn-ghost"),
        ], className="mt-2"),

        html.Div(id="rcm-summary", style={"marginTop": "12px", "color": "#495057"}),
    ],
    body=True,
)

rcm_graph_card = dbc.Card([
    dcc.Graph(id="rcm-graph", config={"displayModeBar": True, "displaylogo": False}, style={"height": "600px"}),
], body=True)

rcm_section = dbc.Container([
    dbc.Row([
        dbc.Col(rcm_controls, md=3),
        dbc.Col(rcm_graph_card, md=9),
    ], className="g-2", align="center")
], fluid=True, style={"maxWidth": "1200px", "padding": "0 12px"})


# ---------- SIDEBAR NAV ----------
sidebar = dbc.Card([
    html.Div("Causal Inference in Regression", className="h5", style={"marginBottom": "8px"}),
    html.Div([
        html.A("Two Faces", href="#two-faces", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("Fundamental Problem", href="#fundamental", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("SLR / MLR & Multicollinearity", href="#slr-mlr", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("Rubin's Causal Model", href="#rubin", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("Simpson's Paradox", href="#simpson", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("Conclusion", href="#conclusion", className="btn btn-outline-primary w-100 mb-2 text-start sidebar-btn"),
        html.A("References", href="#references", className="btn btn-outline-primary w-100 mb-0 text-start sidebar-btn"),
    ])
], body=True, className="toc-card")

# ---------- APP LAYOUT ----------
app.layout = html.Div([
    dcc.Location(id="url"),
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, md=2),
            dbc.Col([
                html.Div(id="top", className="section-anchor"),
                md_header,

                html.Div(id="two-faces", className="section-anchor"),
                md_two_faces,

                html.Div(id="fundamental", className="section-anchor"),
                md_fundamental,

                html.Div(id="slr-mlr", className="section-anchor"),
                md_slr_mlr,
                ovb_section,

                html.Div(id="rubin", className="section-anchor"),
                md_rubin,
                rcm_section,

                html.Div(id="simpson", className="section-anchor"),
                md_simpson,
                md_sim_header,
                simpson_section,

                html.Div(id="conclusion", className="section-anchor"),
                md_conclusion,

                html.Div(id="references", className="section-anchor"),
                md_references,

            ], md=10)
        ], className="content-row")
    ], fluid=True)
])

# ---------- CALLBACKS ----------
@app.callback(
    dash.Output("sim-graph", "figure"),
    dash.Output("slope-summary", "children"),
    dash.Input("num-groups", "value"),
    dash.Input("noise", "value"),
    dash.Input("confound", "value"),
    dash.Input("points-per-group", "value"),
    dash.Input({"type": "beta", "index": ALL}, "value"),
    dash.Input("seed", "value"),
    dash.Input("btn-randomize", "n_clicks"),
    dash.Input("btn-reset", "n_clicks"),
    dash.Input("preset-pos", "n_clicks"),
    dash.Input("preset-mix", "n_clicks"),
    dash.Input("preset-neg", "n_clicks"),
)
def update_graph(n_groups, sigma, confound, points_per_group, beta_values, seed_text, n_rand, n_reset, n_pos, n_mix, n_neg):
    # Build betas from dynamic inputs; fallback to 1.0 if missing/invalid
    betas = []
    try:
        betas = [float(v) if v is not None and str(v) != "" else 1.0 for v in beta_values]
    except Exception:
        pass

    if not isinstance(n_groups, int):
        try:
            n_groups = int(n_groups)
        except Exception:
            n_groups = 2

    if len(betas) != n_groups:
        # pad or trim to match selected n_groups
        if len(betas) < n_groups:
            betas = (betas + [1.0] * n_groups)[:n_groups]
        else:
            betas = betas[:n_groups]

    seed = None
    if str(seed_text).strip() != "":
        try:
            seed = int(str(seed_text).strip())
        except Exception:
            seed = None

    # Apply actions
    ctx = dash.callback_context
    if ctx and ctx.triggered:
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "btn-randomize":
            seed = np.random.randint(0, 10_000_000)
        elif trig == "btn-reset":
            n_groups = 2
            sigma = 1.0
            confound = 4.0
            points_per_group = 200
            betas = [1.0, 1.0]
            seed = None
        elif trig == "preset-pos":
            betas = [1.0] * n_groups
        elif trig == "preset-mix":
            betas = [1.0 if i % 2 == 0 else -1.0 for i in range(n_groups)]
        elif trig == "preset-neg":
            betas = [-1.0] * n_groups

    x, y, groups = generate_simpsons_data(
        n_groups=n_groups,
        points_per_group=int(points_per_group),
        betas=betas,
        sigma=sigma,
        confound_strength=confound,
        seed=seed,
    )
    fig = build_simpsons_figure(x, y, groups)

    group_slopes, agg_slope = compute_slopes(x, y, groups)
    group_parts = [f"{name}: {slope:+.3f}" for name, slope in sorted(group_slopes.items())]
    within_sign = np.sign(np.median(list(group_slopes.values()))) if group_slopes else 0
    paradox = (within_sign != 0) and (np.sign(agg_slope) != within_sign)
    badge = html.Span("Paradox: Yes" if paradox else "Paradox: No",
                      className=f"badge-paradox {'badge-yes' if paradox else 'badge-no'}")
    summary = html.Div([
        html.Span("Slopes ", style={"fontWeight": 600}),
        html.Span("(" + ", ".join(group_parts) + f")  |  Aggregate: {agg_slope:+.3f}  "),
        badge,
    ])

    return fig, summary


@app.callback(
    dash.Output("beta-inputs", "children"),
    dash.Input("num-groups", "value"),
)
def render_beta_inputs(n_groups):
    try:
        n = int(n_groups)
    except Exception:
        n = 2
    rows = []
    for i in range(n):
        rows.append(
            dbc.InputGroup(
                [
                    dbc.InputGroupText(f"β{i+1}"),
                    dbc.InputGroupText("=", className="beta-equals"),
                    dbc.Input(
                        id={"type": "beta", "index": i},
                        type="number",
                        step=0.1,
                        value=1.0,
                        debounce=True,
                    ),
                ],
                className="mb-2 beta-input-group",
            )
        )
    return rows




# ---------- CALLBACKS: OVB / Multicollinearity ----------
@app.callback(
    dash.Output("ovb-graph", "figure"),
    dash.Output("ovb-summary", "children"),
    dash.Input("ovb-n", "value"),
    dash.Input("ovb-sigma", "value"),
    dash.Input("ovb-beta1", "value"),
    dash.Input("ovb-beta2", "value"),
    dash.Input("ovb-rho", "value"),
    dash.Input("ovb-seed", "value"),
    dash.Input("ovb-rand", "n_clicks"),
    dash.Input("ovb-reset", "n_clicks"),
)
def update_ovb(n, sigma, beta1, beta2, rho, seed_text, n_rand, n_reset):
    # Handle seed and reset
    seed = None
    try:
        if str(seed_text).strip() != "":
            seed = int(str(seed_text).strip())
    except Exception:
        seed = None

    ctx = dash.callback_context
    if ctx and ctx.triggered:
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "ovb-rand":
            seed = np.random.randint(0, 10_000_000)
        elif trig == "ovb-reset":
            n = 1000
            sigma = 1.0
            beta1 = 1.0
            beta2 = 1.0
            rho = 0.6
            seed = None

    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    mean = np.array([0.0, 0.0])
    xz = rng.multivariate_normal(mean=mean, cov=cov, size=int(n))
    x = xz[:, 0]
    z = xz[:, 1]
    y = float(beta1) * x + float(beta2) * z + rng.normal(0.0, float(sigma), size=int(n))

    # SLR (omit Z)
    slr_beta, slr_intercept = compute_linear_fit(x, y)

    # MLR (include Z)
    X = np.column_stack([np.ones_like(x), x, z])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    mlr_beta = float(coef[1])

    # Partial residuals for visualization
    # Regress Y on Z
    Z1 = np.column_stack([np.ones_like(z), z])
    coef_yz, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    y_res = y - (coef_yz[0] + coef_yz[1] * z)
    # Regress X on Z
    coef_xz, *_ = np.linalg.lstsq(Z1, x, rcond=None)
    x_res = x - (coef_xz[0] + coef_xz[1] * z)

    # Fit on residuals (should equal mlr_beta)
    pr_beta, pr_intercept = compute_linear_fit(x_res, y_res)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("SLR: Y vs X", "MLR: Partial (Y~X|Z)"))

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=6, opacity=0.6, color="#4e79a7"),
                              name="Data", hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>"), row=1, col=1)
    x_line = np.array([x.min(), x.max()])
    y_line = slr_beta * x_line + slr_intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="SLR fit", line=dict(color="#111", width=3)), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_res, y=y_res, mode="markers", marker=dict(size=6, opacity=0.6, color="#59a14f"),
                              name="Residualized", hovertemplate="X_res=%{x:.2f}<br>Y_res=%{y:.2f}<extra></extra>"), row=1, col=2)
    xr_line = np.array([x_res.min(), x_res.max()])
    yr_line = pr_beta * xr_line + pr_intercept
    fig.add_trace(go.Scatter(x=xr_line, y=yr_line, mode="lines", name="MLR partial fit", line=dict(color="#111", width=3)), row=1, col=2)

    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(family="Inter, Segoe UI, system-ui, -apple-system", size=15),
                      margin=dict(l=40, r=30, t=60, b=40))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    bias_est = slr_beta - float(beta1)
    summary = html.Div([
        html.Span("β̂_SLR ", style={"fontWeight": 600}), html.Span(f"= {slr_beta:+.3f}  "),
        html.Span(" | β̂_MLR ", style={"fontWeight": 600}), html.Span(f"= {mlr_beta:+.3f}  "),
        html.Span(" | ρ(X,Z) ", style={"fontWeight": 600}), html.Span(f"= {float(rho):+.2f}  "),
        html.Br(),
        html.Span("SLR bias vs true β1: ", style={"fontWeight": 600}), html.Span(f"{bias_est:+.3f}"),
    ])

    return fig, summary


# ---------- CALLBACKS: Rubin Causal Model ----------
@app.callback(
    dash.Output("rcm-graph", "figure"),
    dash.Output("rcm-summary", "children"),
    dash.Input("rcm-n", "value"),
    dash.Input("rcm-sigma", "value"),
    dash.Input("rcm-tau", "value"),
    dash.Input("rcm-alpha", "value"),
    dash.Input("rcm-k", "value"),
    dash.Input("rcm-seed", "value"),
    dash.Input("rcm-rand", "n_clicks"),
    dash.Input("rcm-reset", "n_clicks"),
)
def update_rcm(n, sigma, tau, alpha, k, seed_text, n_rand, n_reset):
    # Handle seed and reset
    seed = None
    try:
        if str(seed_text).strip() != "":
            seed = int(str(seed_text).strip())
    except Exception:
        seed = None

    ctx = dash.callback_context
    if ctx and ctx.triggered:
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "rcm-rand":
            seed = np.random.randint(0, 10_000_000)
        elif trig == "rcm-reset":
            n = 2000
            sigma = 1.0
            tau = 1.0
            alpha = 1.0
            k = 2.0
            seed = None

    rng = np.random.default_rng(seed)
    n = int(n)
    x = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, float(sigma), size=n)

    # Potential outcomes
    y0 = 0.0 + float(alpha) * x + eps
    y1 = y0 + float(tau)

    # Assignment with propensity depending on x
    logits = float(k) * x
    e = 1.0 / (1.0 + np.exp(-logits))
    t = rng.binomial(1, p=np.clip(e, 1e-6, 1 - 1e-6))

    y = t * y1 + (1 - t) * y0

    # Estimates
    naive = float(y[t == 1].mean() - y[t == 0].mean()) if (t.sum() > 0 and (n - t.sum()) > 0) else float("nan")
    ipw = float(np.mean(t * y / e - (1 - t) * y / (1 - e)))

    # Plot: scatter and bars
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=("Observed data (color = T)", "ATE estimates"))

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=6, opacity=0.65, color=np.where(t==1, "#e15759", "#4e79a7")),
                              name="Units", hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)

    bars_x = ["True", "Naive", "IPW"]
    bars_y = [float(tau), naive, ipw]
    fig.add_trace(go.Bar(x=bars_x, y=bars_y, marker_color=["#59a14f", "#f28e2b", "#edc948"], showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="ATE", row=1, col=2)

    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(family="Inter, Segoe UI, system-ui, -apple-system", size=15),
                      margin=dict(l=40, r=30, t=60, b=40))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    summary = html.Div([
        html.Span("True ATE τ ", style={"fontWeight": 600}), html.Span(f"= {float(tau):+.3f}  "),
        html.Span(" | Naive diff ", style={"fontWeight": 600}), html.Span(f"= {naive:+.3f}  "),
        html.Span(" | IPW ", style={"fontWeight": 600}), html.Span(f"= {ipw:+.3f}  "),
        html.Br(),
        html.Span("Mean propensity e(X): ", style={"fontWeight": 600}), html.Span(f"{float(e.mean()):.3f}"),
    ])

    return fig, summary


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)
