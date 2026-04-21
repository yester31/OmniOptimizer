"""Training modifier plugins for scripts/train.py.

Each module exposes two functions:

    def apply(yolo: "YOLO", spec: "TrainingSpec") -> None:
        '''Mutate the YOLO wrapper (pruning masks / fake-quant modules /
        sparsity state). Called before ultralytics model.train().'''

    def finalize(yolo: "YOLO", spec: "TrainingSpec", out_pt: "Path") -> None:
        '''Serialize the trained model to out_pt. Called after training.
        prune_24 writes a plain state_dict; modelopt_* use mto.save.'''

Spec reference: docs/superpowers/specs/2026-04-20-qr-training-pipeline-design.md
"""
