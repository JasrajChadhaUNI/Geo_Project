"""Inference-only Unet-Siam-Diff, adapted from Monopoli, Montello & Rossi (2024).
Original implementation: https://github.com/links-ads/igarss-landslide-delineation
Only the supplied no-DEM ResNet-50 architecture is implemented here.
The trained checkpoint contains the encoder AND decoder; this is not new training.
"""

from pathlib import Path
import hashlib
import torch
import segmentation_models_pytorch as smp

BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
CHECKPOINT_SHA256 = "0c16cb4353e2d5e092032f04786c8f81179b3c584d57072419c4daa7f88b79cd"


def load_checkpoint(path):
    """Check the exact supplied weights, build their architecture and load strictly."""
    path = Path(path)
    digest_builder = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            "Checkpoint digest does not match the supplied official weights."
        )
    # The verified checkpoint stores argparse.Namespace metadata, so weights_only=False is required.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["hparams"]
    hp = saved.copy() if isinstance(saved, dict) else vars(saved).copy()
    expected = {
        "MODEL": "unet",
        "ENCODER": "resnet50",
        "FUSION_MODULE": "diff_features",
        "N_CLASSES": 1,
    }
    for key, value in expected.items():
        if hp.get(key) != value:
            raise ValueError(f"Unexpected checkpoint {key}: {hp.get(key)}")
    if hp.get("ADD_DEM") not in (None, False) or list(hp["BANDS"]) != BANDS:
        raise ValueError("Expected the 12-band, no-DEM model.")
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=12, classes=1
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    def forward_pair(before, after, unused_dem=None):
        model.check_input_shape(before)
        model.check_input_shape(after)
        before_features = model.encoder(before)
        after_features = model.encoder(after)
        # The first feature is not consumed by the U-Net decoder. Set it to zero deterministically.
        differences = [torch.zeros_like(before_features[0])]
        # Preserve the trained signed difference: AFTER minus BEFORE (not absolute difference).
        differences += [
            post - pre for pre, post in zip(before_features[1:], after_features[1:])
        ]
        decoded = model.decoder(differences)  # segmentation-models-pytorch 0.5.0
        return model.segmentation_head(decoded)

    model.forward = forward_pair
    return model.cpu().eval(), hp
