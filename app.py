import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Example data for demo (you can remove this)
df = px.data.gapminder().query("year == 2007")

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "My Interactive Blog"

# ---------- BLOG CONTENT SECTION ----------
# You can paste your blog paragraphs here as Markdown.
# Use ###, ####, and **bold/italic** syntax for structure.

blog_content = dcc.Markdown(r"""
# Title of Your Blog Post
*(Subtitle or tagline if you want one)*  

---

## Introduction  
Ask anyone with a high school diploma what they remember and you’ll usually get one of two answers: “the mitochondria is the powerhouse of the cell” or “correlation is not causation.” I’m not here to talk about cellular real estate so today we will be covering the latter. In principle, the phrase is quite self-explanatory: correlation is merely an association of two pieces of data whereas causation occurs when one event directly contributes to the occurrence of the other.

![Correlation vs Causation](image1.png)

There are countless instances of correlated events not having a causal relationship, such as the increase in associate degrees awarded in science technologies and google searches for “avocado toast.” Does this mean that as students developed their engineering skills their entire career was leading to the breakthrough of engineering the perfect avocado toast? Not quite. More of these examples can be found on Tyler Vigen’s project on suspicious correlations. 
 
Though in reverse, we cannot have causation without correlation. This leads us to question the other requirements for causal relationships, at what point can we say the perfection of avocado toast was caused by the increase in associate degrees awarded in science technologies?

Before we get there, it’s important to look at the inferences of regression, specifically, the difference between prediction and causal inference. Prediction involves the comparison of outcomes between different units. An example of this would be comparing the test scores of two students: one who studied for the test and one who did not. Causal inference takes the same unit and examines multiple outcomes. This type of inference addresses the issue of “correlation is not causation” because it fixates on cause-and-effect relationships to draw conclusions beyond statistical association.  

Standard regression is a tool built for prediction, not causation. So to be able to draw causal conclusions from regression models, we must focus on strict assumptions and frameworks. Going forward, we will explore different types of data collection, including the gold standard of randomized experiments and struggles with observational data, and helpful graphical and outcome frameworks we can use.

### The Fundamental Problem
  
Now where exactly is the problem with causal inference? To visualize this, I like to think about the butterfly effect. Imagine you’re a student who realized too late that there’s an exam tomorrow (perhaps for a particular regression class). Your mind is split between two choices:

- Choice 1: Do not study. Get a solid night’s sleep and hope that a rested mind will carry you through it (or you’re just drawing bunnies the entire time).
- Choice 2: Study. Sacrifice sleep, cram all night, and hope the extra studying pays off.

In your head, you can play out both timelines. Maybe in Option A, you bomb the exam, flunk out of your program, and end up juggling on the street for cash. Or in Option B, you ace the exam and the class, impress your professor, and land your dream job on behalf of your academic merit.

But in reality, you only get to live one of those outcomes. In this case, you either sleep or you don't, which determines the test score you receive. The other path is forever unknowable. Introducing the fundamental problem of causal inference: for any individual, we can only observe the outcome of the decision they actually made and not the alternative reality. Ideally, we, as model engineers, want to know how outcomes, test scores, would change due to different treatments, studying done. Regression only provides access to the single outcome of the choice that was made, but the counterfactuals, or the roads not taken, are always missing.
   
### SLR, MLR, and Multicollinearity
In the context of simple linear regression, take the example above of predicting exam scores, Y, from study hours, X1. To build a simple regression model we hypothesize the model to be of the form

$$Y = \beta_0 + \beta_1(\text{Study Hours}) + \epsilon$$

Though since we will never truly know the values of $\beta_0$ and $\beta_1$ we create estimates:

$$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1(\text{Study Hours})$$

Because of the assumptions we’ve made, our interpretation of the model is limited. We say that $\hat{\beta}_1$ estimates the average change in test score for every one-hour increase in study hours. But this is all a measure of association, not causation. The slope tells us what happens when X changes, but not why this occurs. This increase could be the result of something else, maybe students who study more attend class more or have more genuine interest in the subject matter. These hidden causes that affect both the predictor and the outcome are known as confounding variables.

From the perspective of causal inference, regression coefficients identify patterns in data, through direction and strength of relationships, but not the mechanisms that generate them. To move towards causal inference, we shift our focus to the changes in Y resulting from X, rather than the changes in X associated with Y.

However, this doesn’t mean regression coefficients are useless for causal work, we can never prove causation if we have not first identified association. It simply means we look beyond standard errors and t-tests, but rather the assumptions that went into collecting the data.

Now, returning to our model above, let’s add a confounder: a student’s interest in the subject, denoted by Z. This variable affects both study time and performance.

$$Y_i = \beta_0 + \beta_1(X_i) + \beta_2(Z_i) + \epsilon_i$$

If we look back at our originally proposed simpler model, the OLS estimator for $\beta_1$ becomes biased.

The simple regression estimate $\tilde{\beta}_1$ is biased because X (study hours) and the omitted variable Z (interest) are correlated.

$$\tilde{\beta}_1 = \beta_1 + \beta_2 \frac{\text{Cov}(X,Z)}{\text{Var}(X)}$$

This shows us the bias that comes from omitted variables, unless either $\beta_2$ (interest has no effect on exam performance) or Cov(X,Z) = 0 (interest is independent of study hours) then our estimated slope for study hours ( β1^) is biased.  
So the solution here is multiple linear regression (MLR) so we can include all confounding variables.  

The case of Multiple Linear Regression (MLR)  
In the multiple regression case, the variance of the estimated coefficient $\hat{\beta}_1$ is given by:  

$$
\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{(1 - R^2_{(X*Z)}) \cdot n \cdot \text{Var}(X)}
$$

Where  
$R^2_{(X*Z)}$= coefficient of determination from regression X onto Z  
$\sigma^2$ = residual variance  
n = sample size  

As $R^2_{(X*Z)}$ -> 1 (X and Z are almost perfectly collinear), the denominator approaches zero and the $\text{Var}(\hat{\beta}_1)$ explodes.  
Note that they can never be collinear as the variance would be undefined.  

This means that while including Z (interest) corrects bias in theory, it destabilizes the estimate of the causal effect by inflating uncertainty because the variance inflation factor (VIF) increases with multicollinearity.

**Multicollinearity in Prediction vs. Causal Models**

This is where prediction models differ from causal models. With causal models, our overarching goal is to reduce bias so we can make that ultimate causal inference. To achieve this, we must ensure we have an unbiased causal estimator, which means all the relevant confounders are included in the model.

This begs the question of multicollinearity, does this mean that causal models favor multicollinear terms? Not necessarily.

In MLR for prediction, multicollinearity is a numerical issue. if two regressors X and Z are highly correlated, the matrix $X^TX$ becomes nearly singular, and the OLS variance formula blows up:

$$Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}$$

This is often indicative of large standard errors, and unreliable coefficient interpretations. That said, OLS remains unbiased, our predictions just become messier.

Now looking at MLR for causation, we want to know if the variation in X is independent (enough) of confounders. If X and our confounder Z are nearly collinear, then after we control Z we are left with little independent variation in X. Mathematically,

$$\hat{\beta}_1 = \frac{Cov(Y,X|Z)}{Var(X|Z)}$$

If Var(X|Z) is small (because they move together), we don't have as much information to estimate the causal effect. So multicollinearity here results in the loss of the ability to identify causality. 

### Rubin Causal Model

So causal inference boils down to a missing data problem. How do we estimate what we can't see? This question is foundational to the Rubin Causal Model (RCM) which defines causal effects as a comparison of what would have happened to the same units under different treatments. Let's change our example and instead of studying, we've discovered a drug that condenses an entire module's worth of content into one digestible pill. We're still interested in test scores, but we want to see if our drug works.

So let's consider the simple case of a binary treatment, where $T_i$ represents the treatment status of the *i*th unit. $Y_i$ has two potential outcomes:

Let $Y(1)$ = observed outcome of receiving treatment (uses drug and does not study)

And $Y(0)$ = observed outcome if that same unit were in the control group (does not use drug and does not study).

Before the treatment is applied, both $Y(1)$ and $Y(0)$ exist as potentials for every unit (randomization). After intervention, only one of these is observed.

This gives us

$$Y_i=(T_i)Y_i(1) + (1-(T_i))(Y_i(0))$$

The treatment effect for the *i*th unit can be defined as $Y_i(1) - Y_i(0)$ which measures the gain in the outcome variable (test scores) under the assignment to the treatment. Now to statistically overcome the fundamental problem of causal inference, we want to generalize to estimate the average treatment effect rather than an individual one.

$\theta = E\{Y_i(1)\} - E\{Y_i(0)\}$

Which can also be expressed as

$E\{Y_i(1)\} = E\{Y_i(1)|T_i\}$

The **independence assumption** ensures

$\theta = E\{Yi(1)|Ti = 1\} - E\{Yi(0)|Ti = 0\}$

Gives an unbiased estimate for $\theta$

Now to address **strong ignorability**

# Observational Studies, Strong Ignorability, and Randomization

We've addressed the fundamental problem with causal inference, but this makes it tricky to interpret real-world data. Most data is observational where people opt into a treatment. Sicker patients are more likely to take a drug. Now, it is possible to draw causal inferences from observational data, but it requires an assumption, namely, the ignorable treatment assignment assumption. This assumption states that, conditional on observed pre-treatment covariates X, the assignment to treatment is independent of the potential outcomes (Y0, Y1). This means that measuring and controlling all confounding variables, X, that influence treatment decision and outcome, then the treatment assignment can be viewed as random.

As defined by the RCM, to make the assumption of strong ignorability, the following must hold for each i

(i) Unconfoundedness

$Yi(1) - Yi(0) \perp T|Xi$

This is the conditional probability of assignment to treatment given the pretreatment variable. In simple terms, the assignment to treatment (T) is independent of the outcomes after conditioning the observed pre-treatment variables (X). So within a subpopulation defined by the same covariate $(Xi = x)$ the assignment to treatment is as good as random, meaning it is fair to compare both treatment and control group at that level of X.

When this condition holds, we are able to estimate the average treatment effect for a given subpopulation $(\theta(x))$ by comparing the average observed outcomes between treated and controlled units within that group. So what we're building up to mathematically,

$\theta(x) = E\{Yi(1)|Ti = 1, Xi = x\} - E\{Yi(0)|Ti = 0, Xi = x\}$

So once unconfoundedness is satisfied, we can average $\theta(x)$ over all possible X values to get an unbiased estimator of $\theta$, the overall average treatment effect.

(ii) Overlap

$0 < P(Ti = 1|Xi = x) < 1$

We call this the propensity score. In english: for every set of pre-treatment characteristics $X = x$, there must be a non-zero probability that a unit with those characteristics receives the treatment AND that they receive the control. If this assumption were violated for some x value, then units with that characteristics would be found only in the treatment or only in the control group. Then it would be impossible to estimate the potential outcome of the unobserved group.

---

The idea behind all of this is that ignorability implies randomization which gives us the power to say assignment to treatment is independent of potential outcomes.

**Warning! Do NOT control post-treatment variables**

Though it was perfectly fine to take pre-treatment variables into account, variables measured before the treatment were not impacted by the treatment, but measuring a variable after treatment can introduce bias and invalidate the causal estimate, even if we properly randomized! The reason behind this is because we are comparing groups that are fundamentally different. Comparing treatment and control groups on the basis of some outcome score is inherently biased because each individual likely had different potentials long before the experiment took place.

### Simpson's Paradox
To put together all that we’ve discussed so far, let’s look at a real-world example of how regression can be misleading through Simpson’s Paradox. Simpson’s Paradox occurs where an association, or trend, that appears in an entire population is reversed when looking at subpopulations. For instance, a simple linear regression might show that X increases Y, but when we split the data by another variable, Z, each subgroup shows the opposite trend.

How can both of these associations be true? The answer lies in the confounding variables we discussed earlier. Since Z influences both X and Y, it distorts their relationship when combining data. By identifying Z, we’ve found a true causal predictor as the relationship between X and Y fades. Simpson’s Paradox is less of a paradox, and more of a design problem. Ignoring our confounders can lead to a completely different analysis of our data.

## Simpson's Paradox: An Interactive Simulation
Here’s where you transition to your interactive component.
Explain what the user will be able to explore below.

---
    """,
    style={"fontSize": "18px", "lineHeight": "1.7em"}, mathjax=True
)


# ---------- INTERACTIVE DASHBOARD SECTION ----------
# Replace this with your own simulation/graph/callbacks

controls = dbc.Card(
    [
        html.H5("Simulation Controls", className="card-title"),
        html.Label("Example dropdown:"),
        dcc.Dropdown(
            id="continent-dropdown",
            options=[{"label": c, "value": c} for c in df["continent"].unique()],
            value="Asia",
        ),
    ],
    body=True,
)

graph_card = dbc.Card(
    [
        dcc.Graph(id="sim-graph"),
    ],
    body=True,
)

interactive_section = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(graph_card, md=9),
            ],
            align="center",
        )
    ],
    fluid=True,
)

# ---------- BLOG CONTINUATION ----------
blog_followup = dcc.Markdown(
    """

---

## Conclusion  
We began with the age-old warning that “correlation is not causation” where we unpacked this. Along the way, we saw regression is inherently predictive, not causal, as we have to dig deeper to find the “why.” This led us to Rubin’s Causal Model, where we consider two outcomes of each unit, and recognize that only one will ever be observed. This causes us to rely on the assumption of ignorability and independence so we can achieve the gold standard of randomization in real-world data. Overall, causality boils down to a design problem. The most critical part of casual analysis happens long before .fit() is used as the computer will crunch the numbers regardless, but it’s the job of the researcher to design the experiment. If the structure of data collection is not fit for causal inference, no amount of statistical manipulation will change this.

In prediction, we tend to select models through metrics such as Akaike’s Information Criterion (AIC), or adjusted R-squared. But these reward the models that fit the observed data the best, not the best practices of data generation. A model can appear statistically significant, but be causally meaningless if correlation is confused for causation. Meaning, maximizing prediction can sometimes hide the causal bias we’re trying to eliminate. True causal inference relies on theoretical reasoning, not just statistical fit. 

In short, the best predictive model isn’t always the best causal model. Regression helps us describe the world as it is, but causal inference helps us imagine the world that could’ve been. So we end here where we began: correlation may start the story, but only causation writes the plot.



### Key Ideas  
Use bullet points, **bold text**, or even inline equations like `Y = β₀ + β₁X`.

> 💡 You can include callouts, quotes, or tips like this.


---

*Thanks for reading!*
    """,
    style={"fontSize": "18px", "lineHeight": "1.7em"},
)

# ---------- APP LAYOUT ----------
app.layout = dbc.Container(
    [
        html.Br(),
        blog_content,
        html.Br(),
        interactive_section,
        html.Br(),
        blog_followup,
        html.Br(),
        html.Footer(
            "© 2025 Your Name | Built with Plotly Dash",
            style={"textAlign": "center", "marginTop": "40px", "color": "gray"},
        ),
    ],
    fluid=True,
)

# ---------- CALLBACKS ----------
@app.callback(
    dash.Output("sim-graph", "figure"),
    dash.Input("continent-dropdown", "value"),
)
def update_graph(continent):
    filtered = df[df["continent"] == continent]
    fig = px.scatter(
        filtered,
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="country",
        title=f"Life Expectancy vs GDP per Capita ({continent})",
        hover_name="country",
    )
    fig.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    app.run(debug=True)
