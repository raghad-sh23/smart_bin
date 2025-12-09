import os, json, time, argparse, glob
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

IMG_SIZE   = (224, 224)
BATCH      = 32
DATA_DIR   = "dataset"
MODEL_BEST = "smart_bin_mbv2_best.keras"   # default to transfer-learning model
MODEL_FINAL= "smart_bin_mbv2.h5"
CLASSMAP   = "class_map.json"

def parse_class_list(s: str|None):
    if not s: return None
    classes = [c.strip() for c in s.split(",") if c.strip()]
    return classes or None

def load_classmap():
    with open(CLASSMAP, "r") as f:
        return json.load(f)  # {"0":"glass", ...}

def predict_one_image(model, img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    return idx, conf

def cmd_predict_folder(args):
    model = load_model(args.model)
    idx_to_class = load_classmap()
    classes = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(args.folder, "**", p), recursive=True)
    files = sorted(files)
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print("No images found.")
        return

    t0 = time.time()
    rows = []
    for fp in files:
        idx, conf = predict_one_image(model, fp)
        rows.append((os.path.basename(fp), classes[idx], conf))
    dt = time.time() - t0
    ips = len(files)/dt if dt>0 else float('inf')

    print(f"\nPredicted {len(files)} images in {dt:.2f}s  ({ips:.2f} images/sec)\n")
    for name, label, conf in rows:
        print(f"{name:35s} → {label:10s} ({conf*100:.2f}%)")

def cmd_eval(args):
    classes_subset = parse_class_list(args.classes)
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        args.folder, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='categorical', shuffle=False, classes=classes_subset
    )

    model = load_model(args.model)
    y_true = gen.classes

    t0 = time.time()
    probs = model.predict(gen, verbose=0)
    dt = time.time() - t0
    y_pred = np.argmax(probs, axis=1)

    acc = (y_true == y_pred).mean()
    ips = len(y_true)/dt if dt>0 else float('inf')

    inv_map = {v:k for k,v in gen.class_indices.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    print(f"\nTop-1 accuracy: {acc*100:.2f}%")
    print(f"Speed: {ips:.2f} images/sec on {len(y_true)} images")

    # per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        idxs = np.where(y_true == i)[0]
        acc_i = (y_pred[idxs] == i).mean() if len(idxs)>0 else float('nan')
        print(f"  {name:10s}: {acc_i*100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix - Smart Bin (Transfer Learning)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nSaved confusion matrix to confusion_matrix.png")

def main():
    ap = argparse.ArgumentParser(description="Smart Bin v2 — batch predict & evaluate")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("predict-folder", help="Predict many images in a folder (recursive)")
    p1.add_argument("--folder", required=True)
    p1.add_argument("--model", default=MODEL_FINAL)
    p1.add_argument("--limit", type=int, default=30)
    p1.set_defaults(func=cmd_predict_folder)

    p2 = sp.add_parser("eval", help="Evaluate accuracy/speed on labeled folder (subfolders = labels)")
    p2.add_argument("--folder", required=True)
    p2.add_argument("--model", default=MODEL_BEST)
    p2.add_argument("--classes", type=str, default=None, help="e.g. 'glass,metal,paper,plastic'")
    p2.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
