import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import linregress
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import arviz as az
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Advanced Enzyme Kinetics", layout="wide")

st.title("Advanced Bayesian Enzyme Kinetics")
st.markdown("Enhanced parameter estimation with tQSSA and sQSSA models based on Choi et al. 2017")

# Standard Michaelis-Menten ODE (sQSSA)
def mm_ode_sQSSA(y, t, kcat, km, ET):
    """Standard quasi-steady-state approximation"""
    S, P = y
    vmax = kcat * ET
    rate = vmax * S / (km + S + 1e-10)
    return [-rate, rate]

# Total QSSA model (tQSSA) - Choi et al. 2017
def mm_ode_tQSSA(y, t, kcat, km, ET, ST):
    """Total quasi-steady-state approximation - accurate for any ET/ST ratio"""
    S, P = y
    # From Eq. 2 in Choi et al. 2017
    term = ET + km + ST - P
    discriminant = term**2 - 4*ET*(ST - P)
    
    if discriminant < 0:
        discriminant = 0
    
    C = 0.5 * (term - np.sqrt(discriminant))
    rate = kcat * C
    return [-rate, rate]

# Legacy MM ODE for backward compatibility
def mm_ode(y, t, vmax, km):
    S, P = y
    rate = vmax * S / (km + S + 1e-10)
    return [-rate, rate]

# Hill equation ODE with kcat parameterization
def hill_ode(y, t, kcat, k05, n, ET):
    S, P = y
    vmax = kcat * ET
    if S <= 0:
        rate = 0
    else:
        try:
            if n == 1.0:
                rate = vmax * S / (k05 + S)
            else:
                Sn = S**n if S < 1e6 else np.exp(n * np.log(S))
                k05n = k05**n if k05 < 1e6 else np.exp(n * np.log(k05))
                rate = vmax * Sn / (k05n + Sn + 1e-10)
        except:
            rate = 0
    return [-rate, rate]

# Competitive inhibition ODE
def competitive_ode(y, t, vmax, km, ki):
    S, P = y
    # Competitive: I increases apparent Km
    km_app = km * (1 + P/ki)  # P acts as inhibitor
    rate = vmax * S / (km_app + S + 1e-10)
    return [-rate, rate]

# Non-competitive inhibition ODE
def noncompetitive_ode(y, t, vmax, km, ki):
    S, P = y
    # Non-competitive: I decreases apparent Vmax
    vmax_app = vmax / (1 + P/ki)  # P acts as inhibitor
    rate = vmax_app * S / (km + S + 1e-10)
    return [-rate, rate]

# Uncompetitive inhibition ODE
def uncompetitive_ode(y, t, vmax, km, ki):
    S, P = y
    # Uncompetitive: I decreases both Vmax and Km proportionally
    factor = 1 / (1 + P/ki)
    vmax_app = vmax * factor
    km_app = km * factor
    rate = vmax_app * S / (km_app + S + 1e-10)
    return [-rate, rate]

# Substrate inhibition ODE
def substrate_inhibition_ode(y, t, vmax, km, kis):
    S, P = y
    # Substrate inhibition: high [S] inhibits
    rate = vmax * S / (km + S + S**2/kis + 1e-10)
    return [-rate, rate]

# Dynamic PyTensor Op for N curves - MM
class MultiCurveOp(Op):
    def __init__(self, times_list, S0_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.n_curves = len(S0_list)
    
    def make_node(self, vmax, km):
        vmax = pt.as_tensor_variable(vmax)
        km = pt.as_tensor_variable(km)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [vmax, km], outputs)
    
    def perform(self, node, inputs, outputs):
        vmax_val, km_val = inputs
        all_solutions = []
        for times, S0 in zip(self.times_list, self.S0_list):
            sol = odeint(mm_ode, [S0, 0.0], times, args=(float(vmax_val), float(km_val)))
            all_solutions.extend(sol[:, 0])
        outputs[0][0] = np.array(all_solutions, dtype='float64')

# Dynamic PyTensor Op for tQSSA model
class MultiCurveOp_tQSSA(Op):
    """Total QSSA Op - accurate for any ET/ST ratio"""
    def __init__(self, times_list, S0_list, ET_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.ET_list = ET_list
        self.n_curves = len(S0_list)
    
    def make_node(self, kcat, km):
        kcat = pt.as_tensor_variable(kcat)
        km = pt.as_tensor_variable(km)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [kcat, km], outputs)
    
    def perform(self, node, inputs, outputs):
        kcat_val, km_val = inputs
        all_solutions = []
        try:
            for times, S0, ET in zip(self.times_list, self.S0_list, self.ET_list):
                ST = S0  # Total substrate initially
                sol = odeint(mm_ode_tQSSA, [S0, 0.0], times, 
                           args=(float(kcat_val), float(km_val), float(ET), float(ST)))
                if np.any(np.isnan(sol)) or np.any(sol < 0):
                    outputs[0][0] = np.full(sum(len(t) for t in self.times_list), 1e10)
                    return
                all_solutions.extend(sol[:, 0])
            outputs[0][0] = np.array(all_solutions, dtype='float64')
        except:
            outputs[0][0] = np.full(sum(len(t) for t in self.times_list), 1e10)

# Dynamic PyTensor Op for N curves - Hill
class MultiCurveHillOp(Op):
    def __init__(self, times_list, S0_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.n_curves = len(S0_list)
    
    def make_node(self, vmax, k05, n):
        vmax = pt.as_tensor_variable(vmax)
        k05 = pt.as_tensor_variable(k05)
        n = pt.as_tensor_variable(n)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [vmax, k05, n], outputs)
    
    def perform(self, node, inputs, outputs):
        vmax_val, k05_val, n_val = inputs
        all_solutions = []
        for times, S0 in zip(self.times_list, self.S0_list):
            sol = odeint(hill_ode, [S0, 0.0], times, 
                         args=(float(vmax_val), float(k05_val), float(n_val)))
            all_solutions.extend(sol[:, 0])
        outputs[0][0] = np.array(all_solutions, dtype='float64')

# Dynamic PyTensor Op for Competitive Inhibition
class MultiCurveCompetitiveOp(Op):
    def __init__(self, times_list, S0_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.n_curves = len(S0_list)
    
    def make_node(self, vmax, km, ki):
        vmax = pt.as_tensor_variable(vmax)
        km = pt.as_tensor_variable(km)
        ki = pt.as_tensor_variable(ki)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [vmax, km, ki], outputs)
    
    def perform(self, node, inputs, outputs):
        vmax_val, km_val, ki_val = inputs
        all_solutions = []
        for times, S0 in zip(self.times_list, self.S0_list):
            sol = odeint(competitive_ode, [S0, 0.0], times, 
                         args=(float(vmax_val), float(km_val), float(ki_val)))
            all_solutions.extend(sol[:, 0])
        outputs[0][0] = np.array(all_solutions, dtype='float64')

# Dynamic PyTensor Op for Non-competitive Inhibition
class MultiCurveNoncompetitiveOp(Op):
    def __init__(self, times_list, S0_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.n_curves = len(S0_list)
    
    def make_node(self, vmax, km, ki):
        vmax = pt.as_tensor_variable(vmax)
        km = pt.as_tensor_variable(km)
        ki = pt.as_tensor_variable(ki)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [vmax, km, ki], outputs)
    
    def perform(self, node, inputs, outputs):
        vmax_val, km_val, ki_val = inputs
        all_solutions = []
        for times, S0 in zip(self.times_list, self.S0_list):
            sol = odeint(noncompetitive_ode, [S0, 0.0], times, 
                         args=(float(vmax_val), float(km_val), float(ki_val)))
            all_solutions.extend(sol[:, 0])
        outputs[0][0] = np.array(all_solutions, dtype='float64')

# Dynamic PyTensor Op for Substrate Inhibition
class MultiCurveSubstrateInhibitionOp(Op):
    def __init__(self, times_list, S0_list):
        self.times_list = times_list
        self.S0_list = S0_list
        self.n_curves = len(S0_list)
    
    def make_node(self, vmax, km, kis):
        vmax = pt.as_tensor_variable(vmax)
        km = pt.as_tensor_variable(km)
        kis = pt.as_tensor_variable(kis)
        n_total = sum(len(t) for t in self.times_list)
        outputs = [pt.vector()]
        return Apply(self, [vmax, km, kis], outputs)
    
    def perform(self, node, inputs, outputs):
        vmax_val, km_val, kis_val = inputs
        all_solutions = []
        for times, S0 in zip(self.times_list, self.S0_list):
            sol = odeint(substrate_inhibition_ode, [S0, 0.0], times, 
                         args=(float(vmax_val), float(km_val), float(kis_val)))
            all_solutions.extend(sol[:, 0])
        outputs[0][0] = np.array(all_solutions, dtype='float64')

# Sidebar configuration
st.sidebar.header("Configuration")

# Number of curves
n_curves = st.sidebar.slider(
    "Number of substrate concentrations",
    min_value=1,
    max_value=10,
    value=3,
    help="3+ curves recommended for model comparison"
)

# Enzyme concentration input
st.sidebar.markdown("**Enzyme Concentration**")
ET_input = st.sidebar.number_input(
    "[E] (µM)",
    value=0.1,
    min_value=0.001,
    format="%.3f",
    help="Enzyme concentration for all experiments"
)
ET_mM = ET_input / 1000  # Convert to mM for calculations

# Show warnings
if n_curves < 2:
    st.sidebar.warning("Single curve: Vmax/Km correlation will be high. Results uncertain!")
elif n_curves == 2:
    st.sidebar.warning("Two curves: Advanced models may not be identifiable. Consider 3+ curves.")
elif n_curves >= 5:
    st.sidebar.info("With 5+ curves, excellent for model discrimination!")

# Dynamic data input
st.sidebar.header("Data Input")

times_list = []
substrate_list = []
S0_list = []

# Suggested S0 values based on estimated Km
if n_curves >= 2:
    st.sidebar.info("Suggested [S₀] ratios: 0.2×Km, 0.5×Km, 1×Km, 3×Km, 10×Km")

for i in range(n_curves):
    st.sidebar.markdown(f"### Curve {i+1}")
    
    # Suggest S0 values with better defaults
    if i == 0:
        default_S0_text = "Low [S₀] ≈ 0.2×Km"
        default_times = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        default_substrate = "0.025, 0.022, 0.019, 0.017, 0.015, 0.013, 0.011, 0.010, 0.009, 0.008, 0.007"
    elif i == 1:
        default_S0_text = "Medium [S₀] ≈ 1×Km"
        default_times = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        default_substrate = "0.100, 0.089, 0.079, 0.071, 0.063, 0.056, 0.050, 0.045, 0.040, 0.036, 0.032"
    elif i == 2:
        default_S0_text = "High [S₀] ≈ 5×Km"
        default_times = "0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5"
        default_substrate = "0.500, 0.465, 0.432, 0.402, 0.374, 0.348, 0.323, 0.301, 0.280, 0.260, 0.242"
    else:
        default_S0_text = f"Very high [S₀] ≈ {10*(i-1)}×Km"
        default_times = "0, 0.5, 1, 1.5, 2, 2.5, 3"
        default_substrate = f"{1.0*i}, {0.95*i}, {0.90*i}, {0.85*i}, {0.80*i}, {0.75*i}, {0.70*i}"
    
    st.sidebar.caption(default_S0_text)
    
    times_input = st.sidebar.text_area(
        f"Time points",
        default_times,
        key=f"times_{i}",
        height=70
    )
    
    substrate_input = st.sidebar.text_area(
        f"Substrate concentrations",
        default_substrate,
        key=f"substrate_{i}",
        height=70
    )
    
    try:
        times = np.array([float(x.strip()) for x in times_input.split(',')])
        substrate = np.array([float(x.strip()) for x in substrate_input.split(',')])
        S0 = substrate[0]
        
        times_list.append(times)
        substrate_list.append(substrate)
        S0_list.append(S0)
        
    except Exception as e:
        st.error(f"Error parsing curve {i+1}: {e}")
        st.stop()

# Check saturation coverage
S0_array = np.array(S0_list)
S0_ratio = S0_array.max() / S0_array.min()

if S0_ratio < 3:
    st.warning("⚠️ Substrate range is narrow. Consider wider [S₀] spread for better Km estimation.")
elif S0_ratio > 100:
    st.info("💡 Very wide [S₀] range - excellent for parameter estimation!")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Experimental Data & Initial Rates")
    
    # Calculate initial rates
    initial_rates = []
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (times, substrate, S0) in enumerate(zip(times_list, substrate_list, S0_list)):
        # Plot data
        fig.add_trace(go.Scatter(
            x=times, y=substrate,
            mode='markers+lines',
            name=f'[S₀] = {S0:.3f}',
            marker=dict(color=colors[i % len(colors)], size=8),
            line=dict(color=colors[i % len(colors)], dash='dash', width=1)
        ))
        
        # Calculate initial rate (first 20% of time or at least 3 points)
        n_points = max(3, int(len(times) * 0.2))
        if len(times) >= n_points:
            # Linear regression on early points
            slope, intercept, r_value, _, _ = linregress(times[:n_points], substrate[:n_points])
            v0 = -slope  # Convert to positive rate
            initial_rates.append(v0)
            
            # Plot initial rate tangent
            t_extend = times[n_points-1]
            fig.add_trace(go.Scatter(
                x=[0, t_extend],
                y=[S0, S0 + slope * t_extend],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                showlegend=False
            ))
            
            # Add annotation
            fig.add_annotation(
                x=t_extend/2, y=S0 + slope * t_extend/2,
                text=f"v₀={v0:.4f}",
                showarrow=False,
                font=dict(color=colors[i % len(colors)], size=10),
                bgcolor='white'
            )
    
    fig.update_layout(
        title="Progress Curves with Initial Rates",
        xaxis_title="Time",
        yaxis_title="[Substrate]",
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show initial rates table
    if initial_rates:
        st.markdown("**Initial Rate Analysis**")
        rate_df = pd.DataFrame({
            '[S₀] (mM)': S0_list,
            'v₀ (mM/min)': initial_rates,
            'v₀/[S₀]': [v/s for v, s in zip(initial_rates, S0_list)]
        })
        st.dataframe(rate_df.round(4))

with col2:
    st.subheader("Bayesian Estimation")
    
    # Check sQSSA validity
    if 'km_estimate' in locals():
        validity_ratio = ET_mM / (km_estimate + np.mean(S0_list))
        if validity_ratio > 0.1:
            st.warning(f"""
            **sQSSA may be invalid!** ET/(Km+ST) = {validity_ratio:.3f} > 0.1
            
            The standard Michaelis-Menten assumes enzyme ≪ substrate.
            → **tQSSA model recommended** for unbiased estimates.
            """)
            use_tQSSA_default = True
        else:
            st.info(f"sQSSA validity satisfied (ET/(Km+ST) = {validity_ratio:.3f})")
            use_tQSSA_default = False
    else:
        use_tQSSA_default = False
    
    # Auto-estimate priors from initial rates
    if initial_rates and len(initial_rates) >= 2:
        # Estimate Vmax as slightly above highest initial rate
        vmax_guess = max(initial_rates) * 1.2
        
        # Estimate Km using Lineweaver-Burk approximation
        inv_v = [1/v for v in initial_rates]
        inv_s = [1/s for s in S0_list]
        slope_lb, intercept_lb, _, _, _ = linregress(inv_s, inv_v)
        
        vmax_lb = 1 / intercept_lb if intercept_lb > 0 else vmax_guess
        km_lb = slope_lb * vmax_lb if slope_lb > 0 else np.median(S0_list)
        
        # Use more robust estimate
        vmax_estimate = np.mean([vmax_guess, vmax_lb])
        km_estimate = km_lb
        
        st.success(f"""
        **Auto-estimated from initial rates:**
        - Vmax ≈ {vmax_estimate:.4f} mM/min
        - Km ≈ {km_estimate:.4f} mM
        """)
    else:
        vmax_estimate = 0.01
        km_estimate = 0.05
    
    # Prior settings
    st.markdown("**Prior Settings**")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        vmax_prior = st.number_input("Vmax prior mean", value=float(vmax_estimate), format="%.4f", min_value=0.0001)
        vmax_sd = st.number_input("Vmax prior SD", value=0.5, min_value=0.1, max_value=2.0,
                                  help="0.3=confident, 1.0=uncertain")
    
    with col_p2:
        km_prior = st.number_input("Km prior mean", value=float(km_estimate), format="%.4f", min_value=0.0001)
        km_sd = st.number_input("Km prior SD", value=0.5, min_value=0.1, max_value=2.0,
                                help="0.3=confident, 1.0=uncertain")
    
    # Model selection
    st.markdown("**Model Selection**")
    
    # Add tQSSA option
    test_tQSSA = st.checkbox(
        "Test tQSSA model (Choi et al. 2017)",
        value=use_tQSSA_default,
        help="Total QSSA - accurate for any enzyme concentration"
    )
    
    models_to_test = ["Michaelis-Menten"]  # Always test standard MM
    if test_tQSSA:
        models_to_test.append("tQSSA")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # Smart default based on number of curves
        default_hill = n_curves >= 3
        if n_curves < 3:
            help_text = "⚠️ Hill model requires 3+ curves"
        else:
            help_text = "Test for cooperativity"
        
        test_hill = st.checkbox("Test Hill model", 
                               value=default_hill,
                               help=help_text)
        if test_hill:
            models_to_test.append("Hill")
        
        test_substrate_inhib = st.checkbox("Test substrate inhibition",
                                         help="High [S] inhibits enzyme")
        if test_substrate_inhib:
            models_to_test.append("Substrate Inhibition")
    
    with col_m2:
        default_product = n_curves >= 4
        if n_curves < 4:
            help_text = "⚠️ Product inhibition requires 4+ curves"
        else:
            help_text = "Test for product inhibition"
        
        test_product_inhib = st.checkbox("Test product inhibition", 
                                       value=default_product,
                                       help=help_text)
        
        if test_product_inhib:
            inhib_type = st.radio("Inhibition type",
                                ["Competitive", "Non-competitive", "Both"],
                                help="Type of product inhibition to test")
            if inhib_type == "Competitive":
                models_to_test.append("Competitive")
            elif inhib_type == "Non-competitive":
                models_to_test.append("Non-competitive")
            else:
                models_to_test.extend(["Competitive", "Non-competitive"])
    
    # Show warning if too many models for data
    if len(models_to_test) > n_curves:
        st.warning(f"""
        ⚠️ Testing {len(models_to_test)} models with only {n_curves} curves.
        Consider reducing models or adding more data.
        """)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        estimate_sigma = st.checkbox("Estimate noise (σ) from data", value=True,
                                   help="Infer measurement noise level")
        n_samples = st.slider("MCMC samples", 1000, 10000, 5000)
        n_tune = st.slider("Tuning samples", 1000, 5000, 3000)
        target_accept = st.slider("Target acceptance", 0.7, 0.95, 0.90)
        n_chains = st.slider("Number of chains", 2, 4, 4, help="More chains = better diagnostics")
    
    if st.button("🚀 Run Bayesian Estimation", type="primary"):
        with st.spinner(f"Testing {len(models_to_test)} models with {n_curves} curves..."):
            
            # Combine all observations
            substrate_obs = np.concatenate(substrate_list)
            
            # Dictionary to store results
            model_results = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Fit each model
            for idx, model_name in enumerate(models_to_test):
                
                st.write(f"Fitting {model_name}...")
                
                if model_name == "Michaelis-Menten":
                    with pm.Model() as model:
                        # Priors
                        vmax = pm.LogNormal('vmax', mu=np.log(vmax_prior), sigma=vmax_sd)
                        km = pm.LogNormal('km', mu=np.log(km_prior), sigma=km_sd)
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # ODE solutions
                        ode_op = MultiCurveOp(times_list, S0_list)
                        substrate_pred = ode_op(vmax, km)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample
                        trace = pm.sample(
                            draws=n_samples, 
                            tune=n_tune,
                            chains=n_chains,
                            target_accept=target_accept,
                            return_inferencedata=True,
                            idata_kwargs={"log_likelihood": True}
                        )
                    
                    model_results[model_name] = trace
                
                elif model_name == "tQSSA":
                    with pm.Model() as model:
                        # Priors - using kcat instead of vmax
                        kcat_prior = vmax_prior / ET_mM if ET_mM > 0 else vmax_prior
                        kcat = pm.LogNormal('kcat', mu=np.log(kcat_prior), sigma=vmax_sd)
                        km = pm.LogNormal('km', mu=np.log(km_prior), sigma=km_sd)
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # Create ET list for all curves
                        ET_list = [ET_mM] * n_curves
                        
                        # ODE solutions using tQSSA
                        ode_op = MultiCurveOp_tQSSA(times_list, S0_list, ET_list)
                        substrate_pred = ode_op(kcat, km)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample - use DEMetropolisZ for gradient-free sampling
                        try:
                            step = pm.DEMetropolisZ()
                            trace = pm.sample(
                                draws=n_samples, 
                                tune=n_tune,
                                chains=n_chains,
                                step=step,
                                return_inferencedata=True,
                                idata_kwargs={"log_likelihood": True}
                            )
                        except:
                            # Fallback to standard sampler
                            trace = pm.sample(
                                draws=n_samples, 
                                tune=n_tune,
                                chains=n_chains,
                                target_accept=target_accept,
                                return_inferencedata=True,
                                idata_kwargs={"log_likelihood": True}
                            )
                    
                    model_results[model_name] = trace
                
                elif model_name == "Hill":
                    with pm.Model() as model:
                        # Priors
                        vmax = pm.LogNormal('vmax', mu=np.log(vmax_prior), sigma=vmax_sd)
                        k05 = pm.LogNormal('k05', mu=np.log(km_prior), sigma=km_sd)
                        n_hill = pm.TruncatedNormal('n', mu=1.0, sigma=0.2, lower=0.3, upper=3.0)
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # ODE solutions
                        ode_op = MultiCurveHillOp(times_list, S0_list)
                        substrate_pred = ode_op(vmax, k05, n_hill)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample
                        try:
                            trace = pm.sample(
                                draws=n_samples, 
                                tune=n_tune,
                                chains=n_chains,
                                target_accept=target_accept,
                                return_inferencedata=True,
                                idata_kwargs={"log_likelihood": True},
                                init='adapt_diag'
                            )
                            model_results[model_name] = trace
                        except:
                            st.warning(f"Hill model fitting failed. Skipping.")
                
                elif model_name == "Competitive":
                    with pm.Model() as model:
                        # Priors
                        vmax = pm.LogNormal('vmax', mu=np.log(vmax_prior), sigma=vmax_sd)
                        km = pm.LogNormal('km', mu=np.log(km_prior), sigma=km_sd)
                        ki = pm.LogNormal('ki', mu=np.log(km_prior), sigma=1.0)  # Ki prior similar to Km
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # ODE solutions
                        ode_op = MultiCurveCompetitiveOp(times_list, S0_list)
                        substrate_pred = ode_op(vmax, km, ki)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample
                        trace = pm.sample(
                            draws=n_samples, 
                            tune=n_tune,
                            chains=n_chains,
                            target_accept=target_accept,
                            return_inferencedata=True,
                            idata_kwargs={"log_likelihood": True}
                        )
                    
                    model_results[model_name] = trace
                
                elif model_name == "Non-competitive":
                    with pm.Model() as model:
                        # Priors
                        vmax = pm.LogNormal('vmax', mu=np.log(vmax_prior), sigma=vmax_sd)
                        km = pm.LogNormal('km', mu=np.log(km_prior), sigma=km_sd)
                        ki = pm.LogNormal('ki', mu=np.log(km_prior), sigma=1.0)
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # ODE solutions
                        ode_op = MultiCurveNoncompetitiveOp(times_list, S0_list)
                        substrate_pred = ode_op(vmax, km, ki)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample
                        trace = pm.sample(
                            draws=n_samples, 
                            tune=n_tune,
                            chains=n_chains,
                            target_accept=target_accept,
                            return_inferencedata=True,
                            idata_kwargs={"log_likelihood": True}
                        )
                    
                    model_results[model_name] = trace
                
                elif model_name == "Substrate Inhibition":
                    with pm.Model() as model:
                        # Priors
                        vmax = pm.LogNormal('vmax', mu=np.log(vmax_prior), sigma=vmax_sd)
                        km = pm.LogNormal('km', mu=np.log(km_prior), sigma=km_sd)
                        kis = pm.LogNormal('kis', mu=np.log(max(S0_list)*2), sigma=1.0)  # Ki_substrate higher than max [S]
                        
                        if estimate_sigma:
                            sigma = pm.HalfNormal('sigma', sigma=0.05)
                        else:
                            sigma = 0.01
                        
                        # ODE solutions
                        ode_op = MultiCurveSubstrateInhibitionOp(times_list, S0_list)
                        substrate_pred = ode_op(vmax, km, kis)
                        
                        # Likelihood
                        pm.Normal('obs', mu=substrate_pred, sigma=sigma, observed=substrate_obs)
                        
                        # Sample
                        trace = pm.sample(
                            draws=n_samples, 
                            tune=n_tune,
                            chains=n_chains,
                            target_accept=target_accept,
                            return_inferencedata=True,
                            idata_kwargs={"log_likelihood": True}
                        )
                    
                    model_results[model_name] = trace
                
                # Update progress
                progress_bar.progress((idx + 1) / len(models_to_test))
            
            st.session_state['model_results'] = model_results
            st.session_state['models_tested'] = models_to_test
            st.session_state['n_curves'] = n_curves
            st.session_state['estimate_sigma'] = estimate_sigma
            st.session_state['S0_list'] = S0_list
            st.session_state['initial_rates'] = initial_rates
            st.success("✅ All models fitted successfully!")

# Results section
if 'model_results' in st.session_state:
    st.markdown("---")
    st.header("📊 Results")
    
    model_results = st.session_state['model_results']
    models_tested = st.session_state['models_tested']
    estimate_sigma = st.session_state.get('estimate_sigma', True)
    S0_list = st.session_state.get('S0_list', [])
    initial_rates = st.session_state.get('initial_rates', [])
    n_curves = st.session_state.get('n_curves', 0)
    
    # Convert to numpy array for calculations
    S0_array = np.array(S0_list)
    S0_ratio = S0_array.max() / S0_array.min() if len(S0_array) > 0 else 1.0
    
    # Define colors for consistency
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Model comparison
    st.subheader("🔬 Model Comparison")
    
    # Calculate WAIC and LOO for all models
    comparison_data = []
    
    with st.spinner("Calculating model comparison metrics..."):
        for model_name, trace in model_results.items():
            waic = az.waic(trace)
            loo = az.loo(trace)
            
            comparison_data.append({
                'Model': model_name,
                'WAIC': waic.elpd_waic * -2,
                'LOO': loo.elpd_loo * -2,
                'p_waic': waic.p_waic,
                'p_loo': loo.p_loo
            })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('WAIC')
    
    # Add delta values
    comparison_df['ΔWAIC'] = comparison_df['WAIC'] - comparison_df['WAIC'].min()
    comparison_df['ΔLOO'] = comparison_df['LOO'] - comparison_df['LOO'].min()
    
    # Determine best model
    best_model = comparison_df.iloc[0]['Model']
    
    # Color code the best model
    def highlight_best(row):
        if row['Model'] == best_model:
            return ['background-color: lightgreen'] * len(row)
        return [''] * len(row)
    
    styled_df = comparison_df.style.apply(highlight_best, axis=1).format({
        'WAIC': '{:.1f}',
        'LOO': '{:.1f}',
        'ΔWAIC': '{:.1f}',
        'ΔLOO': '{:.1f}',
        'p_waic': '{:.1f}',
        'p_loo': '{:.1f}'
    })
    
    st.dataframe(styled_df)
    
    # Interpretation
    st.info(f"**Best model: {best_model}**")
    
    # Check for significant differences
    if len(comparison_df) > 1:
        second_best_delta = comparison_df.iloc[1]['ΔWAIC']
        if second_best_delta < 2:
            st.warning("⚠️ Top models have similar performance (ΔWAIC < 2). Results may be uncertain.")
        elif second_best_delta < 6:
            st.success("✅ Best model has moderate support (ΔWAIC = {:.1f})".format(second_best_delta))
        else:
            st.success("✅ Best model has strong support (ΔWAIC = {:.1f})".format(second_best_delta))
    
    # Model-specific interpretation
    best_trace = model_results[best_model]
    
    if best_model == "Hill":
        n_mean = best_trace.posterior['n'].mean().values
        n_std = best_trace.posterior['n'].std().values
        if n_mean > 1.3:
            st.warning(f"⚠️ **Positive cooperativity detected** (n = {n_mean:.2f} ± {n_std:.2f})")
        elif n_mean < 0.8:
            st.warning(f"⚠️ **Negative cooperativity detected** (n = {n_mean:.2f} ± {n_std:.2f})")
    
    elif "Competitive" in best_model:
        ki_mean = best_trace.posterior['ki'].mean().values
        st.warning(f"🛑 **Competitive inhibition detected** (Ki = {ki_mean:.3f} mM)")
    
    elif "Non-competitive" in best_model:
        ki_mean = best_trace.posterior['ki'].mean().values
        st.warning(f"🛑 **Non-competitive inhibition detected** (Ki = {ki_mean:.3f} mM)")
    
    elif "Substrate Inhibition" in best_model:
        kis_mean = best_trace.posterior['kis'].mean().values
        st.warning(f"🔴 **Substrate inhibition detected** (Ki,substrate = {kis_mean:.3f} mM)")
    
    # Parameter estimates for best model
    col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
    
    with col_r1:
        st.subheader("Parameter Estimates")
        
        # Get parameter names based on model
        if best_model == "Michaelis-Menten":
            param_names = ['vmax', 'km']
            display_names = ['Vmax', 'Km']
        elif best_model == "Hill":
            param_names = ['vmax', 'k05', 'n']
            display_names = ['Vmax', 'K₀.₅', 'n (Hill)']
        elif best_model in ["Competitive", "Non-competitive"]:
            param_names = ['vmax', 'km', 'ki']
            display_names = ['Vmax', 'Km', 'Ki']
        elif best_model == "Substrate Inhibition":
            param_names = ['vmax', 'km', 'kis']
            display_names = ['Vmax', 'Km', 'Ki,substrate']
        
        summary = az.summary(best_trace, var_names=param_names)
        summary.index = display_names
        
        if estimate_sigma:
            sigma_summary = az.summary(best_trace, var_names=['sigma'])
            sigma_summary.index = ['σ (noise)']
            summary = pd.concat([summary, sigma_summary])
        
        st.dataframe(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
        
        # Store key parameters
        vmax_mean = best_trace.posterior['vmax'].mean().values
        if best_model == "Hill":
            km_mean = best_trace.posterior['k05'].mean().values
        else:
            km_mean = best_trace.posterior['km'].mean().values
    
    with col_r2:
        st.subheader("Diagnostics")
        
        # Calculate diagnostics
        rhat_data = az.rhat(best_trace, var_names=param_names[:2])
        max_rhat = max(float(rhat_data[param_names[0]].values), 
                      float(rhat_data[param_names[1]].values))
        st.metric("Max R̂", f"{max_rhat:.3f}")
        
        ess_data = az.ess(best_trace, var_names=param_names[:2])
        min_ess = min(float(ess_data[param_names[0]].values), 
                     float(ess_data[param_names[1]].values))
        st.metric("Min ESS", f"{min_ess:.0f}")
        
        # Parameter correlation
        vmax_samples = best_trace.posterior['vmax'].values.flatten()
        if best_model == "Hill":
            km_samples = best_trace.posterior['k05'].values.flatten()
        else:
            km_samples = best_trace.posterior['km'].values.flatten()
        corr = np.corrcoef(vmax_samples, km_samples)[0, 1]
        st.metric("Vmax-Km correlation", f"{corr:.3f}")
    
    with col_r3:
        st.subheader("Experimental Design")
        st.metric("Number of curves", n_curves)
        st.metric("[S₀] range", f"{S0_array.min():.3f} - {S0_array.max():.3f}")
        st.metric("[S₀] ratio", f"{S0_ratio:.1f}×")
        
        # Check Km coverage
        km_coverage = sum((s > 0.2*km_mean) & (s < 5*km_mean) for s in S0_list)
        st.metric("Curves near Km", f"{km_coverage}/{n_curves}")
    
    # Enzyme kinetics plot
    st.subheader("Enzyme Kinetics Plot")
    
    if initial_rates:
        fig_kinetics = go.Figure()
        
        # Create smooth S range
        S_min = min(S0_list) * 0.1
        S_max = max(S0_list) * 2
        S_range = np.linspace(S_min, S_max, 200)
        
        # Plot posterior samples (uncertainty)
        n_samples_plot = min(100, len(vmax_samples))
        sample_idx = np.random.choice(len(vmax_samples), n_samples_plot, replace=False)
        
        # Plot based on best model
        for idx in sample_idx:
            if best_model == "Michaelis-Menten":
                v_sample = vmax_samples[idx] * S_range / (km_samples[idx] + S_range)
            elif best_model == "Hill":
                n_samples = best_trace.posterior['n'].values.flatten()
                v_sample = vmax_samples[idx] * (S_range**n_samples[idx]) / (km_samples[idx]**n_samples[idx] + S_range**n_samples[idx])
            elif best_model == "Competitive":
                # For competitive, we show uninhibited curve (P=0)
                v_sample = vmax_samples[idx] * S_range / (km_samples[idx] + S_range)
            elif best_model == "Non-competitive":
                # For non-competitive, we show uninhibited curve (P=0)
                v_sample = vmax_samples[idx] * S_range / (km_samples[idx] + S_range)
            elif best_model == "Substrate Inhibition":
                kis_samples = best_trace.posterior['kis'].values.flatten()
                v_sample = vmax_samples[idx] * S_range / (km_samples[idx] + S_range + S_range**2/kis_samples[idx])
            
            fig_kinetics.add_trace(go.Scatter(
                x=S_range, y=v_sample,
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                opacity=0.2,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Mean curve
        if best_model == "Michaelis-Menten":
            v_mean = vmax_mean * S_range / (km_mean + S_range)
            curve_label = 'Michaelis-Menten fit'
        elif best_model == "Hill":
            n_mean = best_trace.posterior['n'].mean().values
            v_mean = vmax_mean * (S_range**n_mean) / (km_mean**n_mean + S_range**n_mean)
            curve_label = f'Hill fit (n={n_mean:.2f})'
        elif best_model == "Competitive":
            v_mean = vmax_mean * S_range / (km_mean + S_range)
            curve_label = 'Competitive inhibition (P=0)'
        elif best_model == "Non-competitive":
            v_mean = vmax_mean * S_range / (km_mean + S_range)
            curve_label = 'Non-competitive inhibition (P=0)'
        elif best_model == "Substrate Inhibition":
            kis_mean = best_trace.posterior['kis'].mean().values
            v_mean = vmax_mean * S_range / (km_mean + S_range + S_range**2/kis_mean)
            curve_label = 'Substrate inhibition fit'
        
        fig_kinetics.add_trace(go.Scatter(
            x=S_range, y=v_mean,
            mode='lines',
            name=curve_label,
            line=dict(color='black', width=3)
        ))
        
        # Add experimental points
        fig_kinetics.add_trace(go.Scatter(
            x=S0_list,
            y=initial_rates,
            mode='markers',
            name='Measured v₀',
            marker=dict(size=12, color='red', symbol='circle'),
            error_y=dict(
                type='data',
                array=[0.05 * v for v in initial_rates],
                visible=True
            )
        ))
        
        # Add Km indicator
        v_at_km = vmax_mean / 2
        fig_kinetics.add_trace(go.Scatter(
            x=[km_mean, km_mean],
            y=[0, v_at_km],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False
        ))
        
        km_label = "K₀.₅" if best_model == "Hill" else "Km"
        fig_kinetics.add_annotation(
            x=km_mean, y=v_at_km/2,
            text=f"{km_label} = {km_mean:.3f} mM",
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            font=dict(color='green')
        )
        
        # Add Vmax line
        fig_kinetics.add_hline(
            y=vmax_mean, 
            line_dash="dash", 
            line_color="purple",
            annotation_text=f"Vmax = {vmax_mean:.4f} mM/min",
            annotation_position="right"
        )
        
        # Special annotations for inhibition models
        if best_model == "Substrate Inhibition":
            # Mark optimal [S]
            S_opt = np.sqrt(km_mean * kis_mean)
            v_opt = vmax_mean * S_opt / (km_mean + S_opt + S_opt**2/kis_mean)
            fig_kinetics.add_trace(go.Scatter(
                x=[S_opt], y=[v_opt],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='star'),
                name='Optimal [S]'
            ))
            fig_kinetics.add_annotation(
                x=S_opt, y=v_opt,
                text=f"S_opt = {S_opt:.3f} mM",
                showarrow=True,
                arrowhead=2,
                ax=30, ay=-30
            )
        
        # Update layout
        fig_kinetics.update_xaxes(title="[Substrate] (mM)")
        fig_kinetics.update_yaxes(title="Initial Rate v₀ (mM/min)")
        fig_kinetics.update_layout(
            title="Enzyme Kinetics Analysis",
            template='plotly_white',
            legend=dict(x=0.7, y=0.3)
        )
        
        st.plotly_chart(fig_kinetics, use_container_width=True)
    
    # Additional analysis
    st.subheader("Additional Analysis")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        if st.button("📊 Show Posterior Distributions"):
            fig_post = make_subplots(
                rows=1, cols=len(param_names),
                subplot_titles=display_names
            )
            
            for i, param in enumerate(param_names):
                samples = best_trace.posterior[param].values.flatten()
                fig_post.add_trace(go.Histogram(
                    x=samples, nbinsx=50,
                    marker_color=['blue', 'red', 'green', 'orange'][i],
                    opacity=0.7
                ), row=1, col=i+1)
            
            fig_post.update_layout(showlegend=False, template='plotly_white')
            st.plotly_chart(fig_post, use_container_width=True)
    
    with col_a2:
        if st.button("🔍 Compare All Models"):
            # Show fits for all models
            fig_compare = go.Figure()
            
            for model_name, trace in model_results.items():
                # Get parameters for each model
                vmax_m = trace.posterior['vmax'].mean().values
                
                if model_name == "Michaelis-Menten":
                    km_m = trace.posterior['km'].mean().values
                    v_model = vmax_m * S_range / (km_m + S_range)
                elif model_name == "Hill":
                    k05_m = trace.posterior['k05'].mean().values
                    n_m = trace.posterior['n'].mean().values
                    v_model = vmax_m * (S_range**n_m) / (k05_m**n_m + S_range**n_m)
                elif model_name == "Substrate Inhibition":
                    km_m = trace.posterior['km'].mean().values
                    kis_m = trace.posterior['kis'].mean().values
                    v_model = vmax_m * S_range / (km_m + S_range + S_range**2/kis_m)
                else:
                    km_m = trace.posterior['km'].mean().values
                    v_model = vmax_m * S_range / (km_m + S_range)
                
                # Add trace
                fig_compare.add_trace(go.Scatter(
                    x=S_range, y=v_model,
                    mode='lines',
                    name=f"{model_name} (ΔWAIC={comparison_df[comparison_df['Model']==model_name]['ΔWAIC'].values[0]:.1f})",
                    line=dict(width=2)
                ))
            
            # Add data points
            fig_compare.add_trace(go.Scatter(
                x=S0_list,
                y=initial_rates,
                mode='markers',
                name='Data',
                marker=dict(size=10, color='black')
            ))
            
            fig_compare.update_layout(
                title="Model Comparison",
                xaxis_title="[Substrate] (mM)",
                yaxis_title="Initial Rate (mM/min)",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
    
    with col_a3:
        if st.button("💾 Export Results"):
            # Prepare comprehensive results
            export_data = {
                'model_comparison': comparison_df,
                'best_model': best_model,
                'parameters': summary,
                'experimental_data': pd.DataFrame({
                    'S0': S0_list,
                    'v0': initial_rates
                })
            }
            
            # Create multi-sheet Excel file would be nice, but for now CSV
            csv_output = f"# Enzyme Kinetics Analysis Results\n"
            csv_output += f"# Best Model: {best_model}\n\n"
            
            csv_output += "# Model Comparison\n"
            csv_output += comparison_df.to_csv(index=False)
            
            csv_output += "\n# Parameter Estimates\n"
            csv_output += summary.to_csv()
            
            csv_output += "\n# Experimental Data\n"
            csv_output += export_data['experimental_data'].to_csv(index=False)
            
            st.download_button(
                label="Download Results (CSV)",
                data=csv_output,
                file_name="enzyme_kinetics_complete_results.csv",
                mime="text/csv"
            )

# Information section
with st.expander("📚 Understanding the Models"):
    st.markdown("""
    ### Michaelis-Menten Model
    The classic enzyme kinetics model:
    $v_0 = \\frac{V_{max}[S]}{K_m + [S]}$
    
    ### Hill Model (Cooperative Binding)
    For enzymes with multiple binding sites:
    $v_0 = \\frac{V_{max}[S]^n}{K_{0.5}^n + [S]^n}$
    
    ### Competitive Inhibition
    Inhibitor competes with substrate for active site:
    $v_0 = \\frac{V_{max}[S]}{K_m(1 + [I]/K_i) + [S]}$
    
    ### Non-competitive Inhibition
    Inhibitor binds to a different site:
    $v_0 = \\frac{V_{max}[S]}{(K_m + [S])(1 + [I]/K_i)}$
    
    ### Substrate Inhibition
    High substrate concentrations inhibit the enzyme:
    $v_0 = \\frac{V_{max}[S]}{K_m + [S] + [S]^2/K_{is}}$
    
    ### Model Selection
    - **ΔWAIC < 2**: Models are essentially equivalent
    - **ΔWAIC 2-6**: Positive evidence for better model
    - **ΔWAIC > 10**: Strong evidence for better model
    """)

with st.expander("🎯 Tips for Cascade Simulations"):
    st.markdown("""
    ### For Enzyme Cascades
    
    **Why Bayesian is ideal for cascades:**
    - Full posterior distributions propagate uncertainty
    - Product inhibition naturally handled
    - Multiple enzymes can share priors
    
    **Recommended workflow:**
    1. Measure each enzyme individually (3-5 curves)
    2. Test for product/substrate inhibition
    3. Use posterior samples for cascade simulation
    4. Propagate uncertainty through the cascade
    
    **Key considerations:**
    - Product of enzyme 1 = substrate of enzyme 2
    - Product accumulation causes inhibition
    - Flow rates affect local concentrations
    - Temperature/pH may vary in flow reactor
    
    **Export posteriors for simulation:**
    - Use full posterior samples, not just means
    - Monte Carlo through cascade model
    - Optimize flow considering uncertainty
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧪 Advanced Enzyme Kinetics Tool v3.0 | Built with PyMC & Streamlit</p>
    <p>Now with inhibition models for enzyme cascade optimization!</p>
</div>
""", unsafe_allow_html=True)