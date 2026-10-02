# VRI-YOLO11: A Lightweight Model for Automated Wheat Grading

This repository contains the trained weights, custom modules, and model configuration files for **VRI-YOLO11**, a lightweight object detection model for Hard Red Winter (HRW) wheat physical quality grading. VRI-YOLO11 is built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) with targeted architectural modifications that reduce computational cost while preserving detection accuracy.

## Overview

VRI-YOLO11 modifies the YOLO11 architecture through two changes:

- **SEAM attention** in the neck, replacing the C2PSA block with a multi-scale spatial attention module for improved feature discrimination at negligible computational cost.
- **P5 detection head removal**, eliminating the large-object detection scale and its associated bottom-up pathway, since wheat kernels fall within the small-to-medium size range.

Relative to the YOLO11 baseline, VRI-YOLO11 reduces parameters by 39.8.0%, GFLOPs by 17.4%, and model size by 38%, inference latency by 22.9% (1.339ms, 774.6 FPS) while achieving a mAP50 of 95.9% on internal validation and 91.9% on external dataset.

The model detects 10 classes: HRW wheat, four contrasting wheat classes (Durum, Hard Red Spring, Hard White, Soft Red Winter), damaged kernels, shrunken kernels, dockage, stones, and sorghum.

## Repository Structure
├── modules/ # Custom module definitions (drop-in for ultralytics/nn/modules)

├── 11-yaml-files/ # Model configuration YAML files (for ultralytics/cfg/models/11)

├── VRI-YOLO11-Bestpt/ # Trained weights for the proposed VRI-YOLO11 model

├── YOLO11-Bestpt/ # Trained weights for the YOLO11 baseline

└── .gitattributes



> **Note:** The `modules/` and `11-yaml-files/` directories contain the complete set of modules and configurations explored during this study, including experimental variants that were tested but not adopted in the final paper. The configuration used for the published model is **`VRI-YOLO11.yaml`**, and its corresponding trained weights are in **`VRI-YOLO11-Bestpt/`**. Other files are provided for transparency and reproducibility and should be treated as experimental.

## Ablation Study Files

The table below links the architecture configurations behind the ablation study to their training/validation notebooks. Custom modules (`C3k2Ghost`, `SEAM`, etc.) are referenced by the SEAM variants as defined in [`modules/yolo11_fixed_modules.py`](modules/yolo11_fixed_modules.py).

| Variant | Architecture YAML | Notebook | Description |
|---|---|---|---|
| Baseline | [`yolo11.yaml`](11-yaml-files/yolo11.yaml) | [`YOLO11n.ipynb`](Notebooks/YOLO11n.ipynb) | Stock YOLO11 backbone/head, P3-P5 outputs, C2PSA neck |
| NoP5 | [`NoP5.yaml`](11-yaml-files/NoP5.yaml) | — | P5 detection head and its bottom-up path removed; Detect on P3/P4 only |
| SEAM | [`SEAM.yaml`](11-yaml-files/SEAM.yaml) | — | C2PSA replaced with SEAM attention after SPPF; P3-P5 outputs retained |
| NoP5-SEAM | [`NoP5-SEAM.yaml`](11-yaml-files/NoP5-SEAM.yaml) | [`NoP5-SEAM.ipynb`](Notebooks/NoP5-SEAM.ipynb) | Combines the NoP5 head with SEAM attention; basis for the final **VRI-YOLO11** architecture |

> The `NoP5-SEAM.ipynb` notebook loads the architecture by the filename it had before it was renamed to `NoP5-SEAM.yaml` (`Csm-Nop5.yaml`) — kept as-is since it is a working record and its saved cell outputs already reflect that run.

## Requirements

- Python 3.12
- PyTorch 2.x with CUDA support
- Ultralytics 8.3.x

```bash
pip install ultralytics
```

## Quick Start: Inference with Pretrained Weights

`VRI-YOLO11-Bestpt/best.pt` is a pickled model object, not just weights: its SEAM neck layer was defined in a custom module (`modules/yolo11_fixed_modules.py`), so a plain `pip install ultralytics` cannot unpickle it on its own — Python needs to find that class under the exact path the checkpoint was saved with. Register it in memory before loading the checkpoint (no need to modify your Ultralytics installation on disk):

```python
import sys, importlib.util

# Point this at your local copy of modules/yolo11_fixed_modules.py
spec = importlib.util.spec_from_file_location(
    "ultralytics.nn.modules.yolo11_fixed_modules",
    "modules/yolo11_fixed_modules.py",
)
custom_mod = importlib.util.module_from_spec(spec)
sys.modules["ultralytics.nn.modules.yolo11_fixed_modules"] = custom_mod
spec.loader.exec_module(custom_mod)
```

Once that's run, load and use the model as normal:

```python
from ultralytics import YOLO

# Load the trained VRI-YOLO11 model
model = YOLO("VRI-YOLO11-Bestpt/best.pt")

# Run inference on your images
results = model.predict(
    source       = "path/to/images/",
    imgsz        = 1024,
    batch        = 16,
    device       = "cuda:0",
    conf         = 0.001,
    iou          = 0.7,
    agnostic_nms = True,
    max_det      = 500,
    save         = True,
)
```

## Validation

To reproduce validation metrics on a labeled dataset (run the registration snippet from Quick Start first if this is a fresh script/session):

```python
from ultralytics import YOLO

model = YOLO("VRI-YOLO11-Bestpt/best.pt")

metrics = model.val(
    data         = "path/to/data_config.yaml",
    conf         = 0.001,
    iou          = 0.7,
    agnostic_nms = True,
    device       = "cuda:0",
    split        = "val",
    plots        = True,
    project      = "results",
    name         = "vri_yolo11_val",
)
```

Your `data_config.yaml` should follow the standard Ultralytics format, listing image paths and the 10 class names in the correct order.

## Training from Scratch (Optional)

To retrain or modify the architecture, the custom modules must be registered in your local Ultralytics installation:

1. Copy the custom module files into your Ultralytics source tree:
```bash
   cp modules/*.py path/to/ultralytics/nn/modules/
```

2. Register each custom module in `ultralytics/nn/tasks.py` (add to the imports and the module parsing logic) and in `ultralytics/nn/modules/__init__.py` (add to the imports and `__all__` list).

3. Copy the YAML configurations into `ultralytics/cfg/models/11/`.

4. Train:
```python
   from ultralytics import YOLO

   model = YOLO("VRI-YOLO11.yaml")
   model.train(
       data   = "path/to/data_config.yaml",
       epochs = 600,
       imgsz  = 768,
       batch  = 16,
       seed   = 0,
   )
```

## Dataset

The dataset used to train and evaluate VRI-YOLO11 is available here: https://zenodo.org/records/22905434 


## Citation

If you use these weights, code, or the VRI-YOLO11 architecture in your research, please cite:

```bibtex
@article{olagunju2026vriyolo11,
  title   = {VRI-YOLO11: An Optimized YOLO-Based Model for Hard Red Winter Wheat Grading Support},
  author  = {Olusola Olagunju, Doina Caragea, and Yonghui Li},
  journal = {Computers and Electronics in Agriculture},
  year    = {2027},
  doi     = {DOI}
}
```

## License

This repository builds on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0**. Because the custom modules and configurations in this repository extend the Ultralytics framework, this repository is released under the same **AGPL-3.0** license. Please review and comply with its terms.

## Acknowledgments

This work was conducted in the Department of Grain Science and Industry at Kansas State University. The architecture builds on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).


