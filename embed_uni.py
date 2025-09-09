import os, argparse, math
from PIL import Image
import torch
import numpy as np
import timm
from timm.data import resolve_data_config, create_transform
from huggingface_hub import login

def valid_img(p):
    ext = os.path.splitext(p)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

def main():
    parser = argparse.ArgumentParser("Compute UNI(2-h) embeddings on GPU")
    parser.add_argument("--he_dir", type=str, required=True, help="Folder with H&E tiles")
    parser.add_argument("--output_file", type=str, default="he_embeddings.pt",
                        help="Where to save ('.pt' or '.npy')")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Optional resize before model (keeps aspect to square)")
    parser.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="(unused, kept for symmetry)")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token if the model is gated")
    parser.add_argument("--model", type=str, default="UNI2-h",
                        choices=["UNI", "UNI2-h"], help="Which UNI variant to use")
    parser.add_argument("--amp", action="store_true", help="Use fp16 autocast for speed")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading {args.model} weights from HF…")
    timm_kwargs = {}
    if args.model == "UNI2-h":
        timm_kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24,
            init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667*2,
            num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, dynamic_img_size=True
        )
        model_id = "hf-hub:MahmoodLab/UNI2-h"
    else:
        model_id = "hf-hub:MahmoodLab/UNI"

    model = timm.create_model(model_id, pretrained=True, **timm_kwargs).to(device).eval()

    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)

    def preprocess(img: Image.Image):
        if args.img_size:
            img = img.resize((args.img_size, args.img_size), Image.BILINEAR)
        return transform(img)

    files = sorted(
        [os.path.join(args.he_dir, f) for f in os.listdir(args.he_dir)
         if not f.startswith(".") and valid_img(f)]
    )
    if len(files) == 0:
        raise RuntimeError(f"No images found in {args.he_dir}")
    print(f"[INFO] Found {len(files)} images")

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        dvec = model(dummy.to(device)).detach()
        emb_dim = dvec.shape[-1]
    print(f"[INFO] Detected embedding dim: {emb_dim}")

    all_embs = []
    all_names = []
    batch = []
    names = []
    use_amp = args.amp and device.type == "cuda"

    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue
        batch.append(preprocess(img))
        names.append(os.path.basename(p))

        if len(batch) == args.batch_size or i == len(files) - 1:
            x = torch.stack(batch, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        feats = model(x)
                        feats = feats.float()
                else:
                    feats = model(x)
            all_embs.append(feats.cpu())
            all_names.extend(names)
            batch, names = [], []

        if (i + 1) % max(1, args.batch_size * 10) == 0:
            print(f"[INFO] Processed {i+1}/{len(files)}")

    embs = torch.cat(all_embs, dim=0)
    print(f"[INFO] Done. Embeddings shape: {tuple(embs.shape)}")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.output_file.lower().endswith(".npy"):
        np.save(args.output_file, embs.numpy())
        print(f"[SAVE] {embs.shape[0]} embeddings → {args.output_file}")
    else:
        torch.save({"filenames": all_names, "embeddings": embs}, args.output_file)
        print(f"[SAVE] {embs.shape[0]} embeddings (pt dict) → {args.output_file}")

if __name__ == "__main__":
    main()
