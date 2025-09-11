# Quantum Imaging Overlap Analysis Tool

An interactive Streamlit web application for analyzing mechanical and optical mode overlaps in quantum imaging systems, with support for rotation analysis and measurement sweep simulations.

## Features

### Core Functionality
- **Mechanical Mode Analysis**: Square membrane drum modes with configurable kx, ky parameters
- **Optical Mode Analysis**: Hermite-Gaussian (HG) modes with configurable m, n indices
- **Overlap Calculations**: Fast FFT-based overlap mapping between mechanical and optical modes
- **Rotation Analysis**: Illumination mode rotation with configurable angle (-180° to +180°)

### Interactive Controls
- **Grid Settings**: Adjustable membrane size and resolution
- **Mode Parameters**: Real-time adjustment of mechanical and optical mode parameters
- **Visualization**: Multiple colormap options and contour overlays
- **Cut Analysis**: Diagonal, horizontal, vertical, and custom point-to-point cuts

### Visualization Options
- **9 Plot Variants**: Real values, absolute values, and magnitude squared for all three main plots
- **Colormap Selection**: 8 different colormaps (jet, viridis, plasma, inferno, magma, coolwarm, RdYlBu, seismic)
- **Contour Overlays**: Optional contour lines with adjustable levels
- **Interactive Cuts**: Point selection for measurement sweep simulations

### Data Export
- **CSV Export**: Download cut analysis data for further processing
- **Single Point Analysis**: Calculate overlap values at specific coordinates

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
4. View the results in the three main tabs: Real Values, Absolute Values, and Magnitude Squared

### Cut Analysis
1. Enable "Cut Analysis" in the sidebar
2. Select the cut type (Diagonal, Horizontal, Vertical, or Custom)
3. For custom cuts, specify the start and end points
4. View the cut plots and export data as needed

### Single Point Analysis
1. Enter specific x, y coordinates in the "Single Point Analysis" section
2. Click "Calculate Overlap" to get real, absolute, and magnitude squared values

## Technical Details

### Mathematical Background
The application implements:
- Square membrane mechanical modes: φ(x,y) = sin(kx·π·(x+L/2)/L) · sin(ky·π·(y+L/2)/L)
- Hermite-Gaussian optical modes with rotation transformation
- FFT-based convolution for efficient overlap mapping
- Coordinate rotation: x' = x·cos(θ) + y·sin(θ), y' = -x·sin(θ) + y·cos(θ)

### Performance
- Uses Streamlit's `@st.cache_data` for efficient computation caching
- FFT-based overlap calculation for fast real-time updates
- Configurable grid resolution for balance between speed and accuracy

## File Structure
```
Code/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── HG_overlap_v2.ipynb       # Original Jupyter notebook
└── *.csv                     # Data files from notebook analysis
```

## Dependencies
- streamlit >= 1.28.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- pandas >= 2.0.0

## Contributing
This tool is designed for quantum imaging research applications. Feel free to extend the functionality for specific research needs.

## License
This project is part of quantum imaging research at QOM (Quantum Optomechanics Lab).
