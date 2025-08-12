## Data format
### multi camera
*.jpg

### single camera
<camera_name>_*.jpg

## Settings File

```json
{
    "cameras": ["cam2", "cam3", "wide", "cam0", "cam1"], # camera names
    "internal_callibration_type": "single", # single or multi
    "pattern_internal": "chessboard", # chessboard or charuco
    "pattern_external": "chessboard", # chessboard or charuco
    "pattern_size_internal": [11, 8], # number of inner corners
    "pattern_size_external": [11, 8], # number of inner corners
    "pattern_square_size_internal": 0.025, # square size in meters
    "pattern_square_size_external": 0.025 # square size in meters
}
```

**Note**: Image paths are automatically determined based on the `internal_callibration_type`:
- `"single"` → uses `../photos/single_camera` folder
- `"multi"` → uses `../photos/multi_camera` folder

## Intrinsic Calibration with visualisation

```bash
./intrinsic_calibration.sh
```
## Exrinsic Calibration with visualisation

```bash
./extrinsic_calibration.sh
```
