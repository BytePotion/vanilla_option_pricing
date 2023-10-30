import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from qfin.options import BlackScholesCall, BlackScholesPut
from qfin.simulations import MonteCarloCall, MonteCarloPut, GeometricBrownianMotion

def black_scholes(S, K, T, r, sigma, option_type="call"):
     if option_type == "call":
         return BlackScholesCall(S, sigma, K, T, r).price
     else:
         return BlackScholesCall(S, sigma, K, T, r).price - S + K * math.exp(-r * T)
def monte_carlo(S, K, T, r, mu, sigma, steps, paths, option_type="call"):
     if option_type == "call":
         return MonteCarloCall(K, paths, r, S, mu, sigma, T/steps, T).price
     else:
         return MonteCarloPut(K, paths, r, S, mu, sigma, T/steps, T).price

def get_paths(S, T, steps, mu, r, sigma, n):
    paths = []
    for i in range(n):
        paths.append(GeometricBrownianMotion(S, mu, sigma, T/steps, T).simulated_path)
    return paths

st.write('## European Call and Put Option Pricing')

with st.expander('Explanation'):
    st.write("### Black-Scholes Model")
    st.write("The Black-Scholes model provides a closed-form solution for pricing European call and put options.")
    st.latex(r'''
        C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
    ''')
    st.latex(r'''
        P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
    ''')
    
    st.write("### Risk-Neutral Pricing")
    st.write("In the model, we use the risk-free rate \( r \) rather than the drift \( \mu \) to reflect risk-neutral expectations. In a risk-neutral world, the expected return on the underlying asset is the risk-free rate.")
    
    st.write("### Monte Carlo Simulation")
    st.write("The Monte Carlo method simulates various paths for the underlying asset price. For European options, we compute the payoff at maturity \( T \) for each path, average them, and discount back to the present value.")
    st.latex(r'''
        C_{\text{MC}} \approx e^{-rT} \times \frac{1}{N} \sum_{i=1}^{N} \max(S_T(i) - K, 0)
    ''')
    
    st.write("### Error Properties in Monte Carlo Simulation")
    st.write("In a Monte Carlo simulation for option pricing, the standard error (SE) of the estimated option price decreases with the square root of the number of simulation paths \( N \). This is known as the square root rule.")
    st.latex(r'''
        \text{SE} \propto \frac{1}{\sqrt{N}}
    ''')
    st.write("The consequence of this is that the computational effort needed to reduce the error by a factor of \( x \) is \( x^2 \). For instance, to halve the error, you would need to use four times as many paths.")
    st.latex(r'''
        \text{Error}_{\text{new}} = \frac{1}{x} \times \text{Error}_{\text{old}} \implies N_{\text{new}} = x^2 \times N_{\text{old}}
    ''')
    st.write("This quadratic reduction in error is a fundamental property of Monte Carlo methods and informs the choice of the number of simulation paths to achieve a desired level of accuracy.")

    
    st.write("### Link Between Methods")
    st.write("The Monte Carlo simulation and the Black-Scholes model should provide approximately equal pricing for European options under the same parameters.")
    st.latex(r'''
        C_{\text{MC}} \approx C
    ''')
    st.write("### Generally")
    st.write("We assume the underlying asset follows a stochastic differential equation and subsequently derive a pricing functional.")
    st.latex(r'''C = P(\theta) \approx \hat{P}(\theta) = C_{MC}, \theta \in \mathbb{R}^n''')
    st.write("where the parameter set, n, is given by a specific model, e.g.")
    st.latex(r'''\theta = (Black Scholes, Heston, ...)''')
    st.write("This notation becomes useful in machine learning and deep learning frameworks that attempt to learn pricing functionals to accelerate pricing for models that do not exist in closed form.")

# User input in sidebar
with st.sidebar:
    st.header('Input Parameters')
    reset_button = st.button("Reset to Default")
    
    steps = st.slider("Number of Steps", 10, 1000, 100, key="steps")
    r = st.slider("Risk-free Rate", 0.0, 0.1, 0.05, key="r")
    sigma = st.slider("Sigma (σ)", 0.01, 1.0, 0.2, key="sigma")
    paths = st.slider("Number of Paths", 1000, 10000, 2000, key="paths")
    S = st.slider("Initial Asset Price", 50, 200, 100, key="S")
    option_type = st.selectbox("Option Type", ["Call", "Put"], key="option_type")
    st.sidebar.title("A BytePotion App")
    st.image("bytepotion.png", width=200)  # Replace with your image URL or file path
    st.write("This app provides insight into the impact of excess kurtosis on VaR and CVaR for selected Fortune 500 companies.")
    st.write("https://bytepotion.com")
    st.title("Author")
    st.image("roman2.png", width=200)  # Replace with your image URL or file path
    st.write("Roman Paolucci")
    st.write("MSOR Graduate Student @ Columbia University")
    st.write("roman.paolucci@columbia.edu")
    
    if reset_button:
        steps = 100
        r = 0.05
        sigma = 0.2
        paths = 2000
        S = 100

st.write("### Analytical v. Simluation Methods")

# Define a set of strikes
percent_away = np.array([1.3, 1.2, 1.1, 1, .9, .8, .7])
strikes = S * percent_away

# Plot simulated paths
simulated_paths = get_paths(S, T=1, steps=steps, mu=r, r=r, sigma=sigma, n=10)
fig_paths = go.Figure()
for path in simulated_paths:
    fig_paths.add_trace(go.Scatter(x=np.linspace(0, 1, steps), y=path, mode='lines'))
st.plotly_chart(fig_paths)

# Compute prices for the set of strikes
T = 1  # time to maturity

bs_prices = [black_scholes(S, K, T, r, sigma, option_type.lower()) for K in strikes]
mc_prices = [monte_carlo(S, K, T, r, r, sigma, steps, paths, option_type.lower()) for K in strikes]
abs_error = np.abs(np.array(mc_prices) - np.array(bs_prices))

# Pricing and error plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=strikes, y=bs_prices, mode='lines+markers', name='Black-Scholes'))
fig.add_trace(go.Scatter(x=strikes, y=mc_prices, mode='lines+markers', name='Monte Carlo'))
fig.add_trace(go.Scatter(x=strikes, y=abs_error, mode='lines+markers', name='Absolute Error', yaxis='y2'))

fig.update_layout(
    title='Pricing and Absolute Error',
    xaxis_title='Strike',
    yaxis=dict(title='Option Price'),
    yaxis2=dict(title='Absolute Error', overlaying='y', side='right')
)
st.plotly_chart(fig)

st.write("MAE:", sum(abs_error))

st.write("### The Black-Scholes Volatility Surface")

with st.expander('Explanation'):
    # ... [Your previous explanation content] ...
    
    st.write("### Flat Term Structure of Volatility")
    st.write("In our current model, we assume a flat term structure of volatility, meaning that the volatility is constant over different maturities (and of course, strikes or moneynesses). While this is a simplification, it serves as a useful starting point for understanding option pricing.")
    st.latex(r'''
        \sigma(T, K) = \sigma \quad \forall \quad T, K
    ''')
    
    st.write("### Market Implied Volatilities")
    st.write("In practice, option prices in the market are often used to infer implied volatilities. These market implied volatilities often do not match the constant volatility assumption of the Black-Scholes model.")
    
    st.write("### Volatility Surface")
    st.write("Using optimization techniques, one can back out the market's expectation of future volatility at various strikes and maturities, forming what is known as a volatility surface. More specifically, the implied volatility for a given strike and maturity is given by...")
    st.latex(r'''
        \text{argmin}_{\sigma} = \quad |C_{\text{Market}} - C_{\text{BS}}(\sigma)| 
    ''')
    st.write("The volatility surface often exhibits skewness as an empirical fact, indicating the market's complex views on future volatility.")

# 3D Volatility Surface
maturities = np.linspace(0.25, 2, 5)
strikes = np.linspace(50, 150, 5)
strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
vols = np.full_like(strike_grid, sigma)  # flat term structure

fig_3d = go.Figure(data=[go.Surface(z=vols, x=strike_grid, y=maturity_grid)])
fig_3d.update_layout(scene=dict(
                    xaxis_title='Strike',
                    yaxis_title='Maturity',
                    zaxis_title='Volatility'))
st.plotly_chart(fig_3d)