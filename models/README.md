# Models Directory

This directory contains machine learning models for predictive analysis across different race tracks.

## File Structure

### Compressed Archive (for GitHub)
- `models_compressed.tar.gz` (14MB) - Contains all model files compressed
  - Original size: ~51MB
  - Compressed size: 14MB (73% reduction)

### Individual Model Files (ignored by git, auto-extracted on deployment)
- `barber_overall_model.pkl` (9.5MB)
- `barber_features_engineered.parquet` (456KB)
- `sonoma_overall_model.pkl` (23MB)
- `sonoma_features_engineered.parquet` (483KB)
- `road_america_overall_model.pkl` (2.9MB)
- `road_america_features_engineered.parquet` (258KB)
- `circuit_of_the_americas_overall_model.pkl` (14MB)
- `circuit_of_the_americas_features_engineered.parquet` (412KB)
- `model_virginia_international_raceway_overall.pkl` (963KB)
- `virginia_international_raceway_features_engineered.parquet` (298KB)

### Helper Scripts
- `compress_models.sh` - Compress all model files into tar.gz
- `decompress_models.sh` - Extract compressed models

## Usage

### For Local Development

Models are automatically decompressed when you run the application. The `start.sh` script checks for compressed files and extracts them if needed.

### For Deployment

The Docker image includes `models_compressed.tar.gz`. On Cloud Run startup:
1. `start.sh` checks if models are extracted
2. If not, it automatically decompresses `models_compressed.tar.gz`
3. Models are loaded into memory by `load_predictive_models()` in `main.py`

### Manual Compression/Decompression

```bash
# Compress models (if you need to recreate the archive)
cd models
./compress_models.sh

# Decompress models manually
cd models
./decompress_models.sh
```

## Model Information

Each track has:
- **Model file (.pkl)**: Trained scikit-learn model for lap time prediction
- **Feature data (.parquet)**: Engineered features database used for predictions

### Supported Tracks
1. Barber Motorsports Park
2. Sonoma Raceway
3. Road America
4. Circuit of the Americas
5. Virginia International Raceway

## Notes

- Individual `.pkl` and `.parquet` files are ignored by git (too large)
- Only the compressed archive is committed to GitHub
- Models are automatically decompressed on Cloud Run startup
- Minimum 2GB RAM required for model loading

