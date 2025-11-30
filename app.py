"""
NeuroWheel – Real-time Brain-Controlled Wheelchair Simulator
Enhanced Dark Theme UI with Clean Design
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from pathlib import Path
import mne
import warnings
warnings.filterwarnings("ignore")

# Import our pipeline
from src.pipeline import create_pipeline
from src.preprocessing import preprocess_for_eegnet, preprocess_for_mirepnet
from src.utils import detect_eeg_channels, extract_fp1_fp2, load_eeg_data

# Page config
st.set_page_config(
    page_title="NeuroWheel - Brain-Controlled Wheelchair",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
    }
    
    .stApp {
        background: #0f0f1e;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        font-weight: 600;
    }
    
    /* Header Section */
    .neurowheel-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #2d2d5f 0%, #3d3d7f 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid #667eea;
    }
    
    .neurowheel-title {
        font-family: 'Inter', sans-serif;
        font-size: 6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 20%, #ff6bf0 50%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
        margin: 0;
        animation: glow 2s ease-in-out infinite alternate;
        filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.6));
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.8)); }
    }
    
    .subtitle {
        color: #a0a0c0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Upload Area */
    .upload-section {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    /* Command Display */
    .command-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }
    
    .command-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 20px currentColor;
    }
    
    .command-label {
        color: #a0a0c0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5f 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #a0a0c0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.5);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* History Items */
    .history-item {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        color: #e0e0e0;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: transparent;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly Background */
    .js-plotly-plot {
        background: #1a1a2e !important;
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'edf_data' not in st.session_state:
    st.session_state.edf_data = None
if 'channel_map' not in st.session_state:
    st.session_state.channel_map = None
if 'fs' not in st.session_state:
    st.session_state.fs = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_command' not in st.session_state:
    st.session_state.current_command = "STOP"
if 'command_history' not in st.session_state:
    st.session_state.command_history = []
if 'trail_points' not in st.session_state:
    st.session_state.trail_points = []
if 'wheelchair_pos' not in st.session_state:
    st.session_state.wheelchair_pos = [0, 0]
if 'wheelchair_rot' not in st.session_state:
    st.session_state.wheelchair_rot = 0
if 'speed' not in st.session_state:
    st.session_state.speed = 0.0
if 'distance' not in st.session_state:
    st.session_state.distance = 0.0
if 'blink_count' not in st.session_state:
    st.session_state.blink_count = 0
if 'processed_samples' not in st.session_state:
    st.session_state.processed_samples = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None


def create_2d_map(pos_x, pos_y, rotation, trail_points, command):
    """Create dark-themed 2D map with wheelchair position"""
    map_size = 100
    
    fig = go.Figure()
    
    # Dark background
    fig.add_shape(
        type="rect",
        x0=-map_size/2, y0=-map_size/2,
        x1=map_size/2, y1=map_size/2,
        fillcolor="#1a1a2e",
        layer="below",
        line_width=0
    )
    
    # Grid pattern
    grid_spacing = 10
    for i in range(-map_size//2, map_size//2 + 1, grid_spacing):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[-map_size/2, map_size/2],
            mode='lines',
            line=dict(color='#2a2a3e', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[-map_size/2, map_size/2], y=[i, i],
            mode='lines',
            line=dict(color='#2a2a3e', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Roads
    road_width = 3
    road_color = '#3a3a4e'
    
    for y in [-30, 0, 30]:
        fig.add_shape(
            type="rect",
            x0=-map_size/2, y0=y-road_width/2,
            x1=map_size/2, y1=y+road_width/2,
            fillcolor=road_color,
            layer="below",
            line_width=0
        )
    
    for x in [-30, 0, 30]:
        fig.add_shape(
            type="rect",
            x0=x-road_width/2, y0=-map_size/2,
            x1=x+road_width/2, y1=map_size/2,
            fillcolor=road_color,
            layer="below",
            line_width=0
        )
    
    # Path trail with gradient effect
    if len(trail_points) > 1:
        trail_x = [p[0] for p in trail_points]
        trail_y = [p[1] for p in trail_points]
        
        fig.add_trace(go.Scatter(
            x=trail_x, y=trail_y,
            mode='lines',
            line=dict(
                color='#667eea' if command != 'STOP' else '#555566',
                width=4
            ),
            name='Path',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Wheelchair marker
    arrow_length = 2
    arrow_x = arrow_length * np.cos(np.radians(rotation))
    arrow_y = arrow_length * np.sin(np.radians(rotation))
    
    wheelchair_color = '#667eea' if command != 'STOP' else '#777788'
    
    fig.add_trace(go.Scatter(
        x=[pos_x], y=[pos_y],
        mode='markers',
        marker=dict(
            size=20,
            color=wheelchair_color,
            symbol='circle',
            line=dict(width=2, color='white')
        ),
        name='Wheelchair',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=[pos_x, pos_x + arrow_x], y=[pos_y, pos_y + arrow_y],
        mode='lines+markers',
        line=dict(color=wheelchair_color, width=3),
        marker=dict(size=8, color=wheelchair_color),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        xaxis=dict(
            range=[-map_size/2, map_size/2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[-map_size/2, map_size/2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#0f0f1e',
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        dragmode=False
    )
    
    return fig


def process_edf_chunk(data, channel_map, fs, start_idx, chunk_duration_sec=1.0):
    """Process EDF data chunk through models"""
    if data.shape[0] < data.shape[1]:
        n_channels, n_samples = data.shape
        data_2d = data
    else:
        n_samples, n_channels = data.shape
        data_2d = data.T
    
    chunk_size_samples = int(chunk_duration_sec * fs)
    end_idx = min(start_idx + chunk_size_samples, n_samples)
    
    if end_idx <= start_idx:
        return "STOP", 0.0, 0, start_idx
    
    chunk = data_2d[:, start_idx:end_idx]
    blinks = 0
    command = "STOP"
    confidence = 0.0
    
    # EEGNet processing
    try:
        fp_data = extract_fp1_fp2(chunk, channel_map)
        
        if fp_data.shape[0] == 2:
            fp_data = fp_data.T
        
        if fs != 250.0:
            from src.preprocessing import resample_data
            fp_data = resample_data(fp_data, fs, 250.0, axis=0)
        
        if fp_data.shape[0] >= 250:
            fp_chunk = fp_data[:250, :]
            
            eegnet_data = preprocess_for_eegnet(
                fp_chunk, 250.0, target_fs=250.0,
                window_size=250, normalize=True
            )
            
            if st.session_state.pipeline and st.session_state.pipeline.model_loader.eegnet_model:
                try:
                    eegnet_probs, eegnet_preds = st.session_state.pipeline.run_eegnet(
                        eegnet_data[0:1] if len(eegnet_data.shape) == 3 else eegnet_data,
                        threshold=0.5
                    )
                    blinks = int(np.sum(eegnet_preds == 1))
                    st.session_state.blink_count += blinks
                except Exception:
                    pass
    except Exception:
        pass
    
    # MIRepNet processing
    try:
        mi_duration_sec = 4.0
        mi_chunk_size = int(mi_duration_sec * fs)
        mi_end_idx = min(start_idx + mi_chunk_size, n_samples)
        
        if mi_end_idx > start_idx:
            mi_chunk = data_2d[:, start_idx:mi_end_idx]
            
            mirepnet_data = preprocess_for_mirepnet(
                mi_chunk, fs, target_fs=128.0,
                lowcut=4.0, highcut=38.0, target_length=512,
                apply_car=True, normalize=True
            )
            
            if st.session_state.pipeline and st.session_state.pipeline.model_loader.mirepnet_model:
                try:
                    mirepnet_results = st.session_state.pipeline.run_mirepnet(mirepnet_data)
                    command = mirepnet_results['label']
                    confidence = mirepnet_results['confidence']
                except Exception:
                    pass
    except Exception:
        pass
    
    return command, confidence, blinks, end_idx


def update_wheelchair_physics(command, dt=0.1):
    """Update wheelchair position based on command"""
    max_speed = 2.0
    rotation_speed = 45.0
    acceleration = 1.0
    deceleration = 2.0
    
    pos = st.session_state.wheelchair_pos
    rot = st.session_state.wheelchair_rot
    speed = st.session_state.speed
    rot_rad = np.radians(rot)
    
    if command == "Forward":
        speed = min(speed + acceleration * dt, max_speed)
        pos[0] += speed * np.cos(rot_rad) * dt
        pos[1] += speed * np.sin(rot_rad) * dt
    elif command == "Left":
        rot -= rotation_speed * dt
        speed = min(speed + acceleration * dt * 0.5, max_speed * 0.7)
        pos[0] += speed * np.cos(np.radians(rot)) * dt
        pos[1] += speed * np.sin(np.radians(rot)) * dt
    elif command == "Right":
        rot += rotation_speed * dt
        speed = min(speed + acceleration * dt * 0.5, max_speed * 0.7)
        pos[0] += speed * np.cos(np.radians(rot)) * dt
        pos[1] += speed * np.sin(np.radians(rot)) * dt
    else:
        speed = max(speed - deceleration * dt, 0.0)
        if speed > 0:
            pos[0] += speed * np.cos(rot_rad) * dt
            pos[1] += speed * np.sin(rot_rad) * dt
    
    st.session_state.wheelchair_pos = pos
    st.session_state.wheelchair_rot = rot
    st.session_state.speed = speed
    st.session_state.distance += speed * dt
    
    # Always append to trail - never clear the path history
    st.session_state.trail_points.append([pos[0], pos[1]])


def main():
    # Header
    st.markdown("""
    <div class="neurowheel-header">
        <h1 class="neurowheel-title">🧠 NEUROWHEEL</h1>
        <p class="subtitle">Real-time Brain-Controlled Wheelchair Simulator</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        try:
            with st.spinner("Loading AI models..."):
                st.session_state.pipeline = create_pipeline(
                    eegnet_model_path="startle_blink_EEGNet_99_attention.keras",
                    mirepnet_model_path="mirapnet_final_model.pth",
                    device='cpu'
                )
            st.success("✓ Models loaded successfully")
        except Exception as e:
            st.error(f"✗ Error loading models: {e}")
            st.stop()
    
    # File upload
    if st.session_state.edf_data is None:
        st.markdown("""
        <div class="upload-section">
            <h2 style="color: #e0e0e0; margin-bottom: 1rem;">📁 Upload EEG Data</h2>
            <p style="color: #a0a0c0;">Upload an EDF file to begin simulation</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose EDF file",
            type=['edf'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading EDF file..."):
                    temp_path = Path("temp_edf.edf")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    data, channel_names, fs = load_eeg_data(str(temp_path), file_type='edf')
                    channel_map = detect_eeg_channels(data, channel_names)
                    
                    st.session_state.edf_data = data
                    st.session_state.channel_map = channel_map
                    st.session_state.fs = fs
                    st.session_state.processed_samples = 0
                    
                    st.success(f"✓ Loaded: {len(channel_names)} channels @ {fs} Hz")
            except Exception as e:
                st.error(f"✗ Error: {e}")
                st.stop()
    
    # Main simulation
    if st.session_state.edf_data is not None:
        # Start button first
        if not st.session_state.is_running:
            if st.button("▶ Start Simulation", use_container_width=True, key="start_btn"):
                st.session_state.is_running = True
                if st.session_state.start_time is None:
                    st.session_state.start_time = time.time()
                st.rerun()
        
        # Only show map and metrics after starting
        if st.session_state.is_running or st.session_state.processed_samples > 0:
            col1, col2 = st.columns([2.5, 1])
            
            with col1:
                # Process data during simulation
                if st.session_state.is_running:
                    data = st.session_state.edf_data
                    channel_map = st.session_state.channel_map
                    fs = st.session_state.fs
                    start_idx = st.session_state.processed_samples
                    total_samples = data.shape[1] if data.shape[0] < data.shape[1] else data.shape[0]
                    
                    if start_idx < total_samples:
                        command, confidence, blinks, next_idx = process_edf_chunk(
                            data, channel_map, fs, start_idx, chunk_duration_sec=1.0
                        )
                        
                        st.session_state.current_command = command
                        elapsed = time.time() - st.session_state.start_time
                        st.session_state.command_history.append({
                            'time': elapsed,
                            'command': command,
                            'confidence': confidence
                        })
                        if len(st.session_state.command_history) > 10:
                            st.session_state.command_history = st.session_state.command_history[-10:]
                        
                        for _ in range(10):
                            update_wheelchair_physics(command, dt=0.1)
                        
                        st.session_state.processed_samples = next_idx
                    else:
                        st.session_state.is_running = False
                        st.success("✓ Simulation complete!")
                
                # Show map
                fig = create_2d_map(
                    st.session_state.wheelchair_pos[0],
                    st.session_state.wheelchair_pos[1],
                    st.session_state.wheelchair_rot,
                    st.session_state.trail_points,
                    st.session_state.current_command
                )
                
                map_placeholder = st.empty()
                map_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                # Current command
                command_colors = {
                    "Forward": "#667eea",
                    "Left": "#f093fb",
                    "Right": "#4facfe",
                    "STOP": "#777788"
                }
                cmd_color = command_colors.get(st.session_state.current_command, "#667eea")
                
                st.markdown(f"""
                <div class="command-card">
                    <div class="command-label">Current Command</div>
                    <div class="command-text" style="color: {cmd_color};">
                        {st.session_state.current_command}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                latest_conf = st.session_state.command_history[-1]['confidence'] if st.session_state.command_history else 0.0
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
                st.progress(latest_conf)
                st.markdown(f'<div class="metric-value">{latest_conf:.1%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics grid
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Blinks</div>
                        <div class="metric-value">{st.session_state.blink_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Distance</div>
                        <div class="metric-value">{st.session_state.distance:.1f}m</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metrics_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Speed</div>
                        <div class="metric-value">{st.session_state.speed * 3.6:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    elapsed = (time.time() - st.session_state.start_time) if st.session_state.start_time else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Time</div>
                        <div class="metric-value">{elapsed:.1f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress
                if st.session_state.edf_data is not None:
                    total_samples = st.session_state.edf_data.shape[1] if st.session_state.edf_data.shape[0] < st.session_state.edf_data.shape[1] else st.session_state.edf_data.shape[0]
                    progress = st.session_state.processed_samples / total_samples if total_samples > 0 else 0
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Progress</div>', unsafe_allow_html=True)
                    st.progress(progress)
                    st.markdown(f'<div class="metric-value">{progress:.1%}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Command history
                st.markdown("---")
                st.markdown("### 📜 Recent Commands")
                if st.session_state.command_history:
                    for cmd_info in reversed(st.session_state.command_history[-5:]):
                        st.markdown(f"""
                        <div class="history-item">
                            <strong>{cmd_info['time']:.1f}s</strong> → {cmd_info['command']} 
                            <span style="color: #667eea;">({cmd_info['confidence']:.0%})</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color: #777788; text-align: center;">No commands yet</p>', unsafe_allow_html=True)
                
                # Controls
                st.markdown("---")
                col_reset, col_export = st.columns(2)
                
                with col_reset:
                    if st.button("🔄 Reset", use_container_width=True):
                        # Reset all values to initial state
                        st.session_state.current_command = "STOP"
                        st.session_state.command_history = []
                        st.session_state.trail_points = []
                        st.session_state.wheelchair_pos = [0, 0]
                        st.session_state.wheelchair_rot = 0
                        st.session_state.speed = 0.0
                        st.session_state.distance = 0.0
                        st.session_state.blink_count = 0
                        st.session_state.processed_samples = 0
                        st.session_state.start_time = None
                        st.session_state.is_running = False
                        st.rerun()
                
                with col_export:
                    if st.session_state.command_history:
                        results_df = pd.DataFrame(st.session_state.command_history)
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Export",
                            data=csv,
                            file_name="neurowheel_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            # Auto-refresh if running
            if st.session_state.is_running:
                data = st.session_state.edf_data
                total_samples = data.shape[1] if data.shape[0] < data.shape[1] else data.shape[0]
                if st.session_state.processed_samples < total_samples:
                    time.sleep(0.5)
                    st.rerun()


if __name__ == "__main__":
    main()