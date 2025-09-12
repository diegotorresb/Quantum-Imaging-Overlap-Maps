# Quantum Imaging Overlap Analysis Tool

An interactive Streamlit web application for analyzing mechanical and optical mode overlaps in quantum imaging systems, featuring dual visualization modes with Matplotlib and Plotly, rotation analysis, and interactive measurement capabilities.

## Features

### Dual Analysis Interface
- **📊 Matplotlib Analysis**: Traditional static plots with comprehensive cut analysis
- **🎯 Plotly Analysis**: Interactive 3D visualizations 

### Core Functionality
- **Mechanical Mode Analysis**: Square membrane drum modes with configurable kx, ky parameters
- **Optical Mode Analysis**: Hermite-Gaussian (HG) modes with configurable m, n indices
- **Overlap Calculations**: Fast FFT-based overlap mapping between mechanical and optical modes
- **Rotation Analysis**: Illumination mode rotation with configurable angle (-180° to +180°)

### Interactive Controls
- **Grid Settings**: Adjustable membrane size and resolution
- **Mode Parameters**: Real-time adjustment of mechanical and optical mode parameters
- **Visualization**: Multiple colormap options and contour overlays
- **Cut Analysis**: Diagonal and custom point-to-point cuts

### Visualization Options

#### Matplotlib Tab
- **9 Plot Variants**: Real values, absolute values, and magnitude squared for all three main plots
- **Colormap Selection**: 8 different colormaps 
- **Contour Overlays**: Optional contour lines with adjustable levels
- **Custom Cut Sliders**: Square layout with horizontal and vertical sliders for point selection

#### Plotly Tab
- **3D Surface Plots**: Interactive 3D visualization of all three main plots
- **2D Heatmaps**: Interactive heatmaps with hover information
- TODO-**Click-to-Select Cuts**: Click directly on the overlap map to select custom cut points
- **Real-time Updates**: Dynamic cut visualization that updates as you select points
- **Interactive Statistics**: Live metrics and data analysis

### Data Export
- **CSV Export**: Download cut analysis data for further processing
- **Data Export**: Export custom cut data with precise coordinates

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser to `http://localhost:8501`

## Usage

### Basic Analysis
1. Adjust the mechanical mode parameters (kx, ky) in the sidebar
2. Set the optical mode parameters (m, n, σx, σy)
3. Use the rotation angle slider to analyze rotated illumination modes
4. Switch between Matplotlib and Plotly tabs for different visualization styles

### Matplotlib Analysis
1. Select plot value type (Real, Absolute, or Magnitude Squared)
2. Enable cut analysis and choose cut type
3. For custom cuts, use the square slider layout to set precise coordinates
4. Export data as CSV for further analysis

### Plotly Analysis
1. Choose between 2D heatmaps and 3D surface plots
2. Select plot value type (Real, Absolute, or Magnitude Squared)
3. SOON: For custom cuts, click directly on the overlap map to select points

### Single Point Analysis
1. Enter specific x, y coordinates in the "Single Point Analysis" section
2. Click "Calculate Overlap" to get real, absolute, and magnitude squared values

## Technical Details

### Mathematical Background
The application implements:
- Square membrane mechanical modes: φ(x,y) = sin(kx·π·(x+L/2)/L) · sin(ky·π·(y+L/2)/L)
- Hermite-Gaussian optical modes with rotation transformation
- FFT-based convolution for efficient overlap mapping
- Coordinate rotation: x = x·cos(θ) + y·sin(θ), y = -x·sin(θ) + y·cos(θ)

### Performance
- Uses Streamlits `@st.cache_data` for efficient computation caching
- FFT-based overlap calculation for fast real-time updates
- Configurable grid resolution for balance between speed and accuracy
- Interactive Plotly visualizations with optimized rendering

### Interactive Features
- **Click-to-Select**: Click anywhere on the overlap map to set custom cut points
- **3D Navigation**: Rotate, zoom, and pan 3D surface plots
- **Real-time Updates**: All visualizations update instantly with parameter changes
- **Hover Information**: Detailed tooltips show exact values and coordinates

## File Structure
```
Quantum-Imaging-Overlap-Maps/
├── streamlit_app.py          # Main Streamlit application
├── streamlit_app_backup.py   # Backup of the application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── HG_overlap_v2.ipynb       # Original Jupyter notebook
└── .git/                     # Git repository
```

## Dependencies
- streamlit >= 1.28.0
- streamlit-vertical-slider >= 0.0.1
- streamlit-toggle >= 0.0.1
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- plotly >= 5.15.0

## Contributing
This tool is designed for quantum imaging research applications. Feel free to extend the functionality for specific research needs.

## License
This project is part of quantum imaging research at QOM (Quantum Optomechanics Lab).
