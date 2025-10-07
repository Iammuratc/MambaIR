# Installation:

conda env create -f environment_H100.yml

conda activate mambair

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements_H100.txt

pip install -e ./yolov12

### Manually build flash attention (not necessary but faster):
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2 

mkdir -p ./tmp
TMPDIR=./tmp pip install --no-build-isolation --verbose .


# Train a model 

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/custom/train_MambaIRYOLO_2x_xview.yml --launcher pytorch
```

## Configuration file:

- scale: Upscale by this factor; Also update network_g - upscale 
- yolo_weight: Weight of the Yolo-losses compared to the SR-loss
- sr_weight: Weight of the SR-loss compared to the Yolo-losses
- yolo_losses: Weights of Yolo-loss components; See Yolo documentation
- empty_loss: How strong Yolo-loss for images without objects is considered. From 0 to 1


# SR Inference

```
python basicsr/test.py -opt options/test/custom/...
```

Results are in results/<experiment_name>/visualization


# Yolo Testing

Make sure the label folder is next to the generated image folder! If not present, create symlink (ln -s source target/labels)

Depending on the configuration, MambaIR will also add a postfix to the image names. Make sure to disable this or remove the last <num_characters> of the name s.t. the image and label names match:
```
for f in *.png; do mv "$f" "$(echo "$f" | sed -E 's/(.*).{<num_characters>}(\.png)$/\1\2/')"; done
```

For testing, you need to execute the script `yolov12/eval_yolo.py`:

```aiignore
from ultralytics import YOLO

model = YOLO('.../MambaIR/experiments/MambaIRv2_SR_x4/models/yolo_latest.pt').model

metrics = model.val(data=".../MambaIR/train_yolo/data-dota-4x.yaml")
```

The `.yaml` file needs to contain the following:

```aiignore
path: .../MambaIR/results/<experiment_name>/visualization
train: .
val: .

# Classes for DOTA 1.0
names:
  0: plane
  1: ship
  2: vehicle
  3: helicopter
~                     
```

In case of `ModuleNotFoundError: No module named 'yolov12'`, this is related to the apparently still incorrect model saving. Modules are saved as `yolov12.ultralytics...` although it should be  `ultralytics...`
A temporary fix is removing the prefix in `/opt/conda/envs/mambair/lib/python3.10/site-packages/torch/serialization.py`, line 1415:
```
if 'yolov12' in mod_name:
    mod_name = mod_name[8:]
```

To perform object detection on an single image, run
```aiignore
results = model("path/to/image.jpg")
results[0].show()
```

