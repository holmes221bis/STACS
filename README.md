# STACS Framework

## Project Description
This repository implements the **STACS** framework for fishing activity detection from vessel trajectory data. The framework integrates adaptive clustering, feature fusion, and RLE-based post-processing to achieve robust segment-level fishing pattern recognition.

## Requirements
- **Python 3.8**
- Required packages:
  ```text
  numpy==1.21.6
  pandas==1.3.5
  pyproj==3.4.0
  shapely==1.8.2
  xgboost==1.6.2
  scikit-learn==1.0.2
  joblib==1.2.0
  tqdm==4.64.1
  scipy==1.7.3
  ```

## Execution
1. **Configure Paths**  
   Set `BASE_DIR` in the script to match your project root path.

2. **Run Main Script**
   ```bash
   python STACS.py
