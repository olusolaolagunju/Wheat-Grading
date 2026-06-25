# VRI-YOLO11: A Lightweight Model for Automated Wheat Grading

This repository contains the trained weights, custom modules, and model configuration files for **VRI-YOLO11**, a lightweight object detection model for Hard Red Winter (HRW) wheat physical quality grading. VRI-YOLO11 is built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) with targeted architectural modifications that reduce computational cost while preserving detection accuracy.

## Overview

VRI-YOLO11 modifies the YOLO11 architecture through three changes:

- **GhostBottleneck substitution** in the C3K2 backbone modules, replacing standard bottleneck convolutions with cheaper ghost feature generation to reduce parameters and FLOPs.
- **CSMM-SEAM attention** in the neck, replacing the C2PSA block with a multi-scale spatial attention module for improved feature discrimination at negligible computational cost.
- **P5 detection head removal**, eliminating the large-object detection scale and its associated bottom-up pathway, since wheat kernels fall within the small-to-medium size range.

Relative to the YOLO11 baseline, VRI-YOLO11 reduces parameters by 44.0%, GFLOPs by 22.6%, and model size by 35.2%, while achieving a mAP50 of 97.9% on internal validation.

The model detects 10 classes: HRW wheat, four contrasting wheat classes (Durum, Hard Red Spring, Hard White, Soft Red Winter), damaged kernels, shrunken kernels, dockage, stones, and sorghum.

## Repository Structure
├── modules/ # Custom module definitions (drop-in for ultralytics/nn/modules)

├── 11-yaml-files/ # Model configuration YAML files (for ultralytics/cfg/models/11)

├── VRI-YOLO11-Bestpt/ # Trained weights for the proposed VRI-YOLO11 model

├── YOLO11-Bestpt/ # Trained weights for the YOLO11 baseline

└── .gitattributes



> **Note:** The `modules/` and `11-yaml-files/` directories contain the complete set of modules and configurations explored during this study, including experimental variants that were tested but not adopted in the final paper. The configuration used for the published model is **`VRI-YOLO11.yaml`**, and its corresponding trained weights are in **`VRI-YOLO11-Bestpt/`**. Other files are provided for transparency and reproducibility and should be treated as experimental.

## Requirements

- Python 3.12
- PyTorch 2.x with CUDA support
- Ultralytics 8.3.x

```bash
pip install ultralytics
```

## Quick Start: Inference with Pretrained Weights

The simplest way to use VRI-YOLO11 is to run the trained weights directly. No source modifications are required for inference, since the weights file already contains the architecture definition.

```python
from ultralytics import YOLO

# Load the trained VRI-YOLO11 model
model = YOLO("VRI-YOLO11-Bestpt/best.pt")

# Run inference on your images
results = model.predict(
    source       = "path/to/images/",
    conf         = 0.345,
    iou          = 0.30,
    agnostic_nms = True,
    save         = True,
)
```

## Validation

To reproduce validation metrics on a labeled dataset:

```python
from ultralytics import YOLO

model = YOLO("VRI-YOLO11-Bestpt/best.pt")

metrics = model.val(
    data         = "path/to/data_config.yaml",
    conf         = 0.25,
    iou          = 0.30,
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

The dataset used to train and evaluate VRI-YOLO11 is available separately:



It contains 740 annotated images with more than 36,000 labeled instances across the 10 classes listed above.

## Citation

If you use these weights, code, or the VRI-YOLO11 architecture in your research, please cite:

```bibtex
@article{olagunju2026vriyolo11,
  title   = {VRI-YOLO11: An Optimized YOLO-Based Model for Automated Hard Red Winter Wheat Grading},
  author  = {Olusola Olagunju, Doina Caragea, and Yonghui Li},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  doi     = {DOI}
}
```

## License

This repository builds on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0**. Because the custom modules and configurations in this repository extend the Ultralytics framework, this repository is released under the same **AGPL-3.0** license. Please review and comply with its terms.

## Acknowledgments

This work was conducted in the Department of Grain Science and Industry at Kansas State University. The architecture builds on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).


