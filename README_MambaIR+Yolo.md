Create env:
conda env create --file environment.yaml

go to yolov12 -> pip install -e .



git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2  # Latest known stable with H100 support

mkdir -p ./tmp
TMPDIR=./tmp pip install --no-build-isolation --verbose .


python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/custom/train_MambaIRYOLO_2x_xview.yml --launcher pytorch

- Fixed environment: Needs specific cuda and pytorch, need to compile module yourself to work with new hardware
- Created environment files (env_H100 and req_H100)
- Build flash attention for newer GPUs
- Move to other dgx :/
- Still slow; VRAM full, but usage low
- No dist training -> Single process using VRAM
- Use dist training
  - Does not work with current setup
  - Combine models into single module for DDP compatibility
  - Fix DistributedDataParallel setup of MambaIR
  - Old empty (no objects) images solution (set loss to 0) incompatible with DDP: DDP expects all parameters to be used for loss and backpropagation
  - Loss function does not work with empty image -> Change loss function. Added parameter (empty_loss) to change strength of loss if no objects present
- Running with ~1 it/s