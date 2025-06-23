# Understanding Hyperspectral TIFF and NDVI NPY Files

## 1. TIFF File (Input)

- **Shape:** `(64, 64, 125)`
  - First two dimensions (64, 64): Spatial resolution (height × width in pixels).
  - Third dimension (125): Number of spectral bands (wavelengths).
- **Each band:** Captures a different wavelength (from visible to near-infrared)
- **Key Bands:**
  - Band 0: ~490nm (start wavelength)
  - Band 44: ~670nm (Red) → Calculated as (670-490)/4 = 45 (if resolution is 4nm, but your output shows 44)
  - Band 90: ~850nm (NIR) → (850-490)/4 = 90
- **Note:** Band indices may vary by camera. **Verify with your camera specs!**

```
Hyperspectral Cube Structure
┌───────────────────────┐
│ Band 0 (490nm)        │ 64px
│ ...                   │ ↑
│ Band 44 (670nm - Red) │ │
│ ...                   │ │
│ Band 90 (850nm - NIR) │ │
│ ...                   │ ↓
│ Band 124              │
└───────────────────────┘
 64px →
```

---

## 2. NPY File (Output)

- **Shape:** `(64, 64)`
  - 2D array of NDVI values (one per pixel)
- **Data type:** `float32`
- **Typical NDVI value range and interpretation:**

| Value Range | Interpretation           |
|-------------|-------------------------|
| -1.0 to 0.0 | Water/Non-vegetation    |
| 0.0 to 0.2  | Bare soil               |
| 0.2 to 0.5  | Sparse vegetation       |
| 0.5 to 1.0  | Dense healthy vegetation|

---

## 3. Transformation Process

**From TIFF to NDVI NPY:**

```python
# 1. Extract specific bands
red_band = img[:, :, 44]  # 670nm 
nir_band = img[:, :, 90]  # 850nm

# 2. Calculate NDVI (1e-8 avoids division by zero)
ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)  

# 3. Clean data
ndvi = np.clip(ndvi, -1.0, 1.0)  # Force values into valid range
```

**Key Changes:**
- Dimensionality Reduction: (64, 64, 125) → (64, 64) (125 bands → 1 NDVI layer).
- Data Compression:
  - Original TIFF: 64×64×125 = 512,000 values (as uint16 = ~1 MB).
  - NPY: 64×64 = 4,096 values (as float32 = ~16 KB).
  - *Note: File size depends on data type (uint16 for TIFF, float32 for NPY).* 

- **Result:** Single 64×64 grid of NDVI values
- **Compression:** 125 bands → 1 band (NDVI)

---

## 4. Visualization Example

| Hyperspectral Cube (125 bands) | NDVI Map (1 band)         |
|-------------------------------|---------------------------|
| *Schematic: Bands stacked*     | *Schematic: NDVI heatmap* |
| Band 0: 490nm                  | NDVI Values:              |
| ...                            | -1.0 (Water)              |
| Band 44: 670nm (Red)           | ...                       |
| ...                            | 0.8 (Healthy Plants)      |
| Band 90: 850nm (NIR)           |                           |

---

## 5. Practical Implications

- **Storage Efficiency:**
  - NPY file stores only NDVI (4096 values vs 512,000 in TIFF)
- **Analysis Ready:**
  - Usable for vegetation health mapping, ML input, spatial analysis
- **Data Integrity:**
  - `float32` preserves NDVI precision
  - Original 16-bit TIFF values converted during processing