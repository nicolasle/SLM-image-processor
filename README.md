# Scanning Laser Microscopy (SLM) Image Processor

A Python desktop application built with **PyQt6** and **Matplotlib** for loading, visualizing, preprocessing, and analyzing lock-in Scanning Laser Microscopy images.

---

## Data Structure & Format

The application is tailored for lock-in SLM acquisitions where each scan step consists of:
- **`X_PD.tiff` / `Y_PD.tiff`**: Photodiode (PD) in-phase ($X$) and quadrature ($Y$) channels.
- **`X_PR.tiff` / `Y_PR.tiff`**: Photoresponse (PR) in-phase ($X$) and quadrature ($Y$) channels.
- **`*.log` coordinate files**:
  - **Line 1**: X mirror voltages (horizontal axis / image columns, e.g. $[-0.4, +0.4]\text{ V}$).
  - **Line 2**: Y mirror voltages (vertical axis / image rows, e.g. $[-0.7, +0.7]\text{ V}$).
- **Step / Run Folders**: Multi-run datasets organized into subfolders (e.g., `0`, `69`, `98`, `114`, `121`).

---

## Key Features

### 1. Flexible Multi-Channel Architecture
- **Dynamic Image Discovery**: Scans and loads **any** TIFF image files present in the folder, rather than requiring hardcoded PR/PD channels.
- **Automatic Lock-in Pairing**: Automatically detects matching in-phase ($X$) and quadrature ($Y$) pairs (e.g. `X_PD` & `Y_PD`, `X_PR` & `Y_PR`, or `*_X` & `*_Y`) and computes:
  - Amplitude: $R = \sqrt{X^2 + Y^2}$
  - Phase: $\theta = \text{atan2}(Y, X)$
  - Phase-Rotated: $X'(\phi) = X \cos\phi + Y \sin\phi$ and $Y'(\phi) = -X \sin\phi + Y \cos\phi$
- **Channel Manager Dialog** (`Ctrl+M` or Tools menu):
  - **Hierarchical Subfolder Tree**: Organized by Dataset, with dedicated collapsible subfolders for **Raw Channels** (direct TIFF image scans on disk) and **Derived Channels** (further grouped into Lock-in Pairs, Ratios, Differences, and Custom quantities).
  - **Multi-Dataset Support**: Seamlessly displays both Harmonics Domain and Time Domain datasets when transformed.
  - **Live Filter**: Instant search bar to filter channels across all subfolders.
  - **Create Lock-in Pair**: Select any two channels to act as $X$ and $Y$ and automatically generate $R$, $\theta$, and rotated components $X', Y'$.
  - **Create Ratio Channel**: Compute $A / B$ (e.g. signal normalized by laser power or reflection).
  - **Create Difference Channel**: Compute $A - B$ (e.g. background subtraction or differential contrast).

### 2. Multi-Layer Overlay & Blending (Any Channel on Any Channel)
- **Arbitrary Layer Selection**: Choose **ANY** available channel as the Bottom (Base) layer and **ANY** channel as the Top (Overlay) layer.
- **Independent Colormaps & Contrast**: Set independent colormaps and contrast limits for each layer (e.g. grayscale for baseline, inferno/turbo for active signal).
- **Dynamic Alpha Slider**: Instant real-time slider and spinbox ($0\%$ to $100\%$) to smoothly blend the top layer onto the base layer.
- **Cbar Min α=0 → Max α=1 (Transparent Baseline)**: Checkbox that dynamically maps overlay opacity to signal intensity. Background values at or below the colorbar minimum are completely transparent ($\alpha=0$), clearly showing the photodiode reflection / crystal structure underneath, while peak signal values are opaque ($\alpha=1$) to highlight only meaningful response.
- **Dedicated Overlay Tab**: Full-screen overlay workspace with quick controls and synchronized dual line cuts.
- **Dual 1D Line Profile**: In overlay mode, the line profile canvas plots both the overlay signal (left axis) and the base baseline (right twin axis) along the exact same spatial cut line!

### 3. Lock-in Phase Optimization
- **Interactive Phase Slider**: Adjust $\phi \in [-180^\circ, +180^\circ]$ with real-time image updates.
- **Auto-Phase Algorithm**: Computes the optimal phase angle $\phi_0$ to align the lock-in signal into the in-phase channel.
- **Auto-Phase All Runs**: Batch-computes the optimal phase across all scan steps.

### 3. Image Leveling & Preprocessing Pipeline
- **Plane Leveling**:
  - Subtract 1st order 2D plane tilt ($z = ax + by + c$) via least-squares
  - Subtract 2nd order polynomial surface ($z = ax^2 + by^2 + cxy + dx + ey + f$)
- **Line-by-Line Subtraction**:
  - Subtract row mean / median
  - Subtract row linear slope
  - Subtract column mean / median
- **Spatial Filters**:
  - Gaussian blur (custom $\sigma$)
  - Median filter ($3\times3$, $5\times5$, $7\times7$)

### 4. Coordinates & Physical Distance Calibration
- Toggle display between **Mirror Volts (V)** and **Physical Distance ($\mu\text{m}$)**.
- **Objective Pre-sets from `LSM.py`**:
  - **50x**: $C_x = 62.167\ \mu\text{m/V}$, $C_y = 64.654\ \mu\text{m/V}$
  - **100x**: $C_x = 29.722\ \mu\text{m/V}$, $C_y = 31.550\ \mu\text{m/V}$
  - **20x**: $C_x = 155.393\ \mu\text{m/V}$, $C_y = 163.001\ \mu\text{m/V}$
  - **10x**: $C_x = 308.088\ \mu\text{m/V}$, $C_y = 320.422\ \mu\text{m/V}$
  - **5x**: $C_x = 621.576\ \mu\text{m/V}$, $C_y = 643.971\ \mu\text{m/V}$
  - **Custom**: User-editable spinboxes for any setup.
- Invert Y-axis option (toggle origin between lower and upper).

### 5. Interactive Inspection Tools
- **Crosshair Inspection**: Click any pixel to inspect real-time coordinates, pixel indices, and signal value.
- **1D Line Profile Cut & Laser Beam $1/e^2$ Size Fitting**:
  - Draw arbitrary cut line across the 2D image, or use quick Horizontal / Vertical mid-cuts.
  - Interactive 1D profile plot showing signal vs distance along the cut with peak/min/mean readouts.
  - **⚡ Fit Erf (Beam $1/e^2$)**: One-click fitting of the error function ($f(x) = y_0 + m(x-x_0) + A \cdot \text{erf}(\sqrt{2}(x-x_0)/w)$) to knife-edge step transitions.
  - **Beam Metrics in Current Units ($V$ or $\mu\text{m}$)**:
    - $1/e^2$ beam radius: $w$
    - $1/e^2$ beam diameter: $2w$
    - Full Width at Half Maximum: $\text{FWHM} = \sqrt{2\ln 2}\, w \approx 1.1774\, w$
    - Beam center position: $x_0$
    - $10\%-90\%$ knife-edge width ($1.2816\, w$)
    - Goodness of fit ($R^2$, RMSE)
    - If in Volts, automatically displays the calibrated micron size equivalent based on the active microscope objective.
  - **Visual Overlays**: Fitted Erf curve, vertical beam center line, and shaded $1/e^2$ diameter span overlaid directly on the canvas.
  - **Dual-View & Gaussian Profile Dialog (`ℹ Details...`)**:
    - Select channel to fit (e.g. active signal vs photodiode reflection baseline).
    - Toggle linear background drift slope ($m \cdot x$).
    - Restrict fit to specific $X$ sub-range.
    - Inspect the physical Gaussian laser beam intensity profile $I(x) = \frac{d}{dx}\text{Erf}(x)$ with $1/e^2$ and FWHM markers.
    - Copy formatted report to clipboard or export fitted curves to CSV.
  - **Auto-fit Toggle**: Automatically refits and updates beam parameters whenever a new line cut is drawn.
- **ROI Selection**:
  - Drag a rectangular region of interest to inspect live metrics: Mean, Std Dev, Min, Max, Peak-to-Peak, and Pixel Count.

### 6. Multi-View & Series Analysis
- **Dual View Mode**: Side-by-side comparison of two channels (e.g. Photodiode $R_{PD}$ baseline vs Photoresponse $R_{PR}$) with optional synchronized zoom & pan.
- **Run Series Trend**: Automatically plots ROI mean across all loaded run steps (`0`, `69`, `98`, `114`, `121`). Click any point on the trend plot to jump directly to that scan step.

### 7. Smart Dataset Auto-Detection
- **Multi-folder hierarchy**: Folders containing subdirectories (e.g. `0/`, `69/`, `98/`, etc.).
- **Flat Harmonic Series**: Large single folders containing hundreds of sequential scans (e.g. `Harm1` to `Harm300` or `Step0` to `StepN`).
- **Chopper / Lock-in frequency extraction**: Parses lock-in reference frequency from metadata filenames (e.g. `fchop180Hz` $\to 180\text{ Hz}$).
- **Instant lazy loading**: Scans and indexes hundreds of files in $< 10\text{ ms}$; full-resolution images are loaded into memory on-demand.

### 8. Inverse Hermitian Fourier Transform (Harmonic $\to$ Time Domain)
- When harmonic acquisitions are detected, click **"Transform to Time Domain (IFFT)"** in the toolbar or menu.
- Configure:
  - Chopper / Lock-in base frequency ($f_{\text{LIA}}$ in Hz)
  - Time discretization ($N$ points per cycle $T = 1/f_{\text{LIA}}$)
  - Complex phasor mapping ($F_{\text{PR}} = Y_n - iX_n$, $F_{\text{PD}} = -Y_n + iX_n$)
  - Phase shift correction $\phi_{\text{corr}}$
- High-performance `irfft` computation converts lock-in Fourier components directly into real physical time evolution $s(t)$ across the modulation cycle.
- Seamlessly toggle back and forth between Harmonic and Time domains.

### 9. Export Capabilities
- Export 2D image as 32-bit floating point TIFF, publication-quality PNG, or PDF figure.
- Export 1D line profile to CSV.
- Export Run Series trend to CSV.
- Export Batch Summary table (all run steps with ROI statistics and auto-phases) to CSV.

---

## Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/slm-image-processor.git
cd slm-image-processor
pip install -r requirements.txt
```

---

## How to Run

You have multiple convenient ways to launch the application:

1. **Desktop Shortcut**: Double-click **`SLM Image Processor`** on your Desktop.
2. **Folder Shortcut**: Double-click **`SLM Image Processor.lnk`** directly in this folder.
3. **Batch Launcher**: Double-click **`run_app.bat`**.
4. **Command Line**:
   ```bash
   python main.py
   ```
   *(Automatically detects and opens `./SLM_images` if present).*

   Or specify a custom dataset directory:
   ```bash
   python main.py "path/to/data_30K"
   ```

To run the automated verification test suite:
```bash
python test_app.py
python test_hermitian_timedomain.py
python test_image_centering.py
```
