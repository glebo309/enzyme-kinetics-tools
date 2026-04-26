#!/usr/bin/env python3
"""
Simple Michaelis-Menten Fitter
Copy-paste Excel data and fit enzyme kinetics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import dash
from dash import dcc, html, Input, Output, State
import io

app = dash.Dash(__name__)
app.title = "Michaelis-Menten Fitter"


def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten equation: v = Vmax * S / (Km + S)"""
    return Vmax * S / (Km + S)


def fit_mm(substrate_conc, velocity):
    """Fit data to Michaelis-Menten equation"""
    try:
        # Initial guesses
        Vmax_guess = np.max(velocity) * 1.2
        Km_guess = substrate_conc[np.argmin(np.abs(velocity - Vmax_guess/2))]
        
        # Fit
        popt, pcov = curve_fit(
            michaelis_menten, 
            substrate_conc, 
            velocity,
            p0=[Vmax_guess, Km_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        
        Vmax, Km = popt
        perr = np.sqrt(np.diag(pcov))
        
        # R-squared
        residuals = velocity - michaelis_menten(substrate_conc, Vmax, Km)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((velocity - np.mean(velocity))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return Vmax, Km, perr[0], perr[1], r_squared
    except Exception as e:
        return None, None, None, None, None


def calculate_velocity(concentrations, times):
    """Calculate initial velocity from concentration vs time data"""
    try:
        # Simple linear regression for initial slope
        velocities = []
        for i in range(len(concentrations)):
            if i == 0:
                velocities.append(0)
            else:
                dt = times[i] - times[0]
                dc = concentrations[i] - concentrations[0]
                velocities.append(abs(dc / dt) if dt > 0 else 0)
        return np.mean([v for v in velocities if v > 0])
    except:
        return None


app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Michaelis-Menten Fitter", 
                style={'margin': '0', 'fontSize': '24px', 'fontWeight': '600'}),
        html.P("Paste Excel data and fit enzyme kinetics", 
               style={'margin': '5px 0 0 0', 'fontSize': '12px', 'opacity': '0.9'})
    ], style={
        'textAlign': 'center', 
        'padding': '20px', 
        'backgroundColor': '#2196F3', 
        'color': 'white'
    }),
    
    # Instructions
    html.Div([
        html.H3("How to use:", style={'fontSize': '14px', 'marginBottom': '10px'}),
        html.Ol([
            html.Li("Copy data from Excel (at least 2 columns: substrate concentration and velocity)"),
            html.Li("Paste into the box below"),
            html.Li("Select which columns to use"),
            html.Li("Click 'Fit Data' to see results")
        ], style={'fontSize': '12px', 'marginLeft': '20px'}),
        html.P([
            "Example format: First column = [S] (substrate), Second column = v (velocity). ",
            "Or use [S], time, and product concentration to calculate velocity."
        ], style={'fontSize': '11px', 'fontStyle': 'italic', 'color': '#666', 'marginTop': '10px'})
    ], style={
        'maxWidth': '900px', 
        'margin': '20px auto', 
        'padding': '15px', 
        'backgroundColor': '#FFF8E0',
        'borderRadius': '8px',
        'border': '1px solid #FFD54F'
    }),
    
    # Main content
    html.Div([
        # Left panel - Data input
        html.Div([
            html.H3("1. Paste Your Data", 
                    style={'fontSize': '14px', 'marginBottom': '10px', 'fontWeight': '600'}),
            
            dcc.Textarea(
                id='paste-area',
                placeholder='Paste Excel data here (Ctrl+V)...\n\nExample:\n0.5\t10.2\n1.0\t18.5\n2.0\t28.3\n5.0\t38.1\n10.0\t42.5',
                style={
                    'width': '100%',
                    'height': '200px',
                    'fontSize': '11px',
                    'fontFamily': 'monospace',
                    'padding': '10px',
                    'borderRadius': '4px',
                    'border': '1px solid #CCC',
                    'resize': 'vertical'
                }
            ),
            
            html.Div(id='parse-status', style={'marginTop': '10px', 'fontSize': '11px'}),
            
            # Column selection
            html.Div([
                html.H3("2. Select Columns", 
                        style={'fontSize': '14px', 'margin': '20px 0 10px 0', 'fontWeight': '600'}),
                
                html.Div([
                    html.Label("Substrate Concentration [S]:", 
                               style={'fontSize': '12px', 'fontWeight': '600', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='substrate-column',
                        placeholder='Select column...',
                        style={'fontSize': '11px'}
                    ),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Velocity (v):", 
                               style={'fontSize': '12px', 'fontWeight': '600', 'display': 'block', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='velocity-column',
                        placeholder='Select column...',
                        style={'fontSize': '11px'}
                    ),
                ], style={'marginBottom': '15px'}),
                
                html.Button(
                    "Fit Data",
                    id='fit-button',
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'fontSize': '14px',
                        'fontWeight': '600',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'marginTop': '10px'
                    }
                ),
            ], id='column-selector', style={'display': 'none'}),
            
        ], style={
            'flex': '1',
            'padding': '20px',
            'borderRight': '1px solid #DDD',
            'minWidth': '350px'
        }),
        
        # Right panel - Results
        html.Div([
            html.H3("Results", 
                    style={'fontSize': '14px', 'marginBottom': '15px', 'fontWeight': '600'}),
            
            html.Div(id='results-display', 
                     children=html.P("Paste data and fit to see results", 
                                   style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '12px'})),
            
            dcc.Graph(id='plot', style={'marginTop': '20px'}, config={'displayModeBar': True}),
            
        ], style={
            'flex': '2',
            'padding': '20px',
            'minWidth': '500px'
        }),
        
    ], style={
        'display': 'flex',
        'maxWidth': '1400px',
        'margin': '0 auto',
        'backgroundColor': 'white',
        'minHeight': '500px'
    }),
    
    # Data store
    dcc.Store(id='data-store'),
    
], style={
    'backgroundColor': '#F5F5F5',
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
})


@app.callback(
    [Output('data-store', 'data'),
     Output('parse-status', 'children'),
     Output('column-selector', 'style'),
     Output('substrate-column', 'options'),
     Output('velocity-column', 'options')],
    Input('paste-area', 'value'),
    prevent_initial_call=True
)
def parse_pasted_data(text):
    """Parse pasted text into dataframe"""
    if not text or not text.strip():
        return None, "", {'display': 'none'}, [], []
    
    try:
        # Try to parse as TSV (tab-separated, from Excel)
        df = pd.read_csv(io.StringIO(text), sep='\t', header=None)
        
        # If only one column, try comma or space separation
        if len(df.columns) == 1:
            df = pd.read_csv(io.StringIO(text), sep=r'[,\s]+', header=None, engine='python')
        
        # Clean up - remove empty rows/columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Try to convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with any NaN
        df = df.dropna()
        
        if len(df) < 3:
            return None, html.Span("Need at least 3 data points", style={'color': '#f44336'}), {'display': 'none'}, [], []
        
        if len(df.columns) < 2:
            return None, html.Span("Need at least 2 columns (substrate and velocity)", style={'color': '#f44336'}), {'display': 'none'}, [], []
        
        # Create column options
        column_options = [{'label': f'Column {i+1}', 'value': i} for i in range(len(df.columns))]
        
        status = html.Span(
            f"Parsed {len(df)} rows × {len(df.columns)} columns",
            style={'color': '#4CAF50', 'fontWeight': '600'}
        )
        
        return df.to_json(orient='split'), status, {'display': 'block'}, column_options, column_options
        
    except Exception as e:
        return None, html.Span(f"Error parsing data: {str(e)}", style={'color': '#f44336'}), {'display': 'none'}, [], []


@app.callback(
    [Output('results-display', 'children'),
     Output('plot', 'figure')],
    Input('fit-button', 'n_clicks'),
    [State('data-store', 'data'),
     State('substrate-column', 'value'),
     State('velocity-column', 'value')],
    prevent_initial_call=True
)
def fit_and_plot(n_clicks, data_json, substrate_col, velocity_col):
    """Fit Michaelis-Menten and display results"""
    
    if not data_json:
        return "No data available", go.Figure()
    
    if substrate_col is None or velocity_col is None:
        return html.P("Please select both substrate and velocity columns", 
                     style={'color': '#f44336'}), go.Figure()
    
    # Load data
    df = pd.read_json(io.StringIO(data_json), orient='split')
    
    substrate = df.iloc[:, substrate_col].values
    velocity = df.iloc[:, velocity_col].values
    
    # Sort by substrate concentration
    sort_idx = np.argsort(substrate)
    substrate = substrate[sort_idx]
    velocity = velocity[sort_idx]
    
    # Fit
    Vmax, Km, Vmax_err, Km_err, r_squared = fit_mm(substrate, velocity)
    
    if Vmax is None:
        return html.P("Fitting failed - check your data", 
                     style={'color': '#f44336'}), go.Figure()
    
    # Results display
    results = html.Div([
        html.H4("Fitted Parameters:", style={'fontSize': '13px', 'marginBottom': '10px', 'fontWeight': '600'}),
        html.Div([
            html.Div([
                html.Span("Vmax = ", style={'fontWeight': '600'}),
                html.Span(f"{Vmax:.4f} ± {Vmax_err:.4f}", style={'fontFamily': 'monospace'}),
            ], style={'marginBottom': '8px', 'fontSize': '12px'}),
            html.Div([
                html.Span("Km = ", style={'fontWeight': '600'}),
                html.Span(f"{Km:.4f} ± {Km_err:.4f}", style={'fontFamily': 'monospace'}),
            ], style={'marginBottom': '8px', 'fontSize': '12px'}),
            html.Div([
                html.Span("R² = ", style={'fontWeight': '600'}),
                html.Span(f"{r_squared:.4f}", style={'fontFamily': 'monospace'}),
            ], style={'marginBottom': '8px', 'fontSize': '12px'}),
        ], style={
            'padding': '15px',
            'backgroundColor': '#E8F5E9',
            'borderRadius': '6px',
            'border': '1px solid #4CAF50'
        }),
        
        html.H4("Interpretation:", style={'fontSize': '13px', 'margin': '20px 0 10px 0', 'fontWeight': '600'}),
        html.Ul([
            html.Li(f"Vmax = {Vmax:.4f}: Maximum reaction velocity at saturating substrate", 
                   style={'fontSize': '11px', 'marginBottom': '5px'}),
            html.Li(f"Km = {Km:.4f}: Substrate concentration at half Vmax (affinity)", 
                   style={'fontSize': '11px', 'marginBottom': '5px'}),
            html.Li(f"R² = {r_squared:.4f}: Goodness of fit {'(excellent)' if r_squared > 0.95 else '(good)' if r_squared > 0.9 else '(fair)'}", 
                   style={'fontSize': '11px'}),
        ], style={'marginLeft': '20px', 'color': '#555'}),
    ])
    
    # Create plot
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=substrate,
        y=velocity,
        mode='markers',
        name='Data',
        marker=dict(size=10, color='#2196F3', line=dict(color='black', width=1))
    ))
    
    # Fitted curve
    S_fit = np.linspace(0, max(substrate) * 1.1, 200)
    v_fit = michaelis_menten(S_fit, Vmax, Km)
    
    fig.add_trace(go.Scatter(
        x=S_fit,
        y=v_fit,
        mode='lines',
        name='Fit',
        line=dict(color='#f44336', width=2.5)
    ))
    
    # Add Km and Vmax lines
    fig.add_hline(y=Vmax, line_dash="dash", line_color="gray", 
                  annotation_text=f"Vmax = {Vmax:.3f}", annotation_position="right")
    fig.add_hline(y=Vmax/2, line_dash="dot", line_color="gray",
                  annotation_text=f"Vmax/2", annotation_position="right")
    fig.add_vline(x=Km, line_dash="dot", line_color="gray",
                  annotation_text=f"Km = {Km:.3f}", annotation_position="top")
    
    # Styling
    fig.update_layout(
        title=f"Michaelis-Menten Fit (R² = {r_squared:.4f})",
        xaxis_title="[Substrate]",
        yaxis_title="Velocity",
        font=dict(size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1),
        height=500,
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#E0E0E0',
        showline=True,
        linecolor='black',
        linewidth=1.5,
        mirror=True,
        zeroline=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#E0E0E0',
        showline=True,
        linecolor='black',
        linewidth=1.5,
        mirror=True,
        zeroline=False
    )
    
    return results, fig


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MICHAELIS-MENTEN FITTER")
    print("="*70)
    print("\nSimple enzyme kinetics fitting tool")
    print("\nHow to use:")
    print("  1. Copy data from Excel (at least 2 columns)")
    print("  2. Paste into the text box")
    print("  3. Select substrate and velocity columns")
    print("  4. Click 'Fit Data'")
    print("\nThe tool will:")
    print("  - Fit v = Vmax * [S] / (Km + [S])")
    print("  - Show Vmax, Km, and R²")
    print("  - Plot data with fitted curve")
    print("\nhttp://localhost:8070")
    print("="*70 + "\n")
    
    app.run(debug=False, port=8070)