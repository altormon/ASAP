import os
import cv2
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from F4_ASAP_Model import StomataNet
from ultralytics import YOLO

### STOMATA DENSITY ###

INPUT_DIR       = "./Input"
RESULTS_DIR     = "./Results"
CROPS_DIR       = os.path.join(RESULTS_DIR, "Stomata")
os.makedirs(CROPS_DIR, exist_ok=True)
YOLO_WEIGHTS    = "./Arabidopsis_40X_Stomata_Density/runs/detect/train/weights/best.pt"
AP_CONF_THRES   = 0.5    ## aperture threshold
DEN_CONF_THRES  = 0.295   ## density threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_ap  = YOLO(YOLO_WEIGHTS)
yolo_den = YOLO(YOLO_WEIGHTS)

def compute_area_mm2(width, height):
    if (width, height) == (1920, 1080):
        return (321 * 180) / 1000
    if (width, height) == (1600, 1200):
        return (268 * 201) / 1000
    return None

density_rows = []
crop_rows    = []
for img_name in sorted(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(INPUT_DIR, img_name)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    res_ap  = yolo_ap(img_rgb, conf=AP_CONF_THRES)[0]
    boxes_ap = res_ap.boxes.xyxy.cpu().numpy()
    res_den = yolo_den(img_rgb, conf=DEN_CONF_THRES)[0]
    boxes_den = res_den.boxes.xyxy.cpu().numpy()
    area_mm2 = compute_area_mm2(w, h)
    count    = len(boxes_den)
    density  = count / area_mm2 if area_mm2 else None
    density_rows.append([img_name, count, area_mm2, density])
    for idx, (x1, y1, x2, y2) in enumerate(boxes_ap, start=1):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        crop = img_rgb[y1:y2, x1:x2]
        crop_fname = f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg"
        crop_path  = os.path.join(CROPS_DIR, crop_fname)
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_rows.append([img_name, idx, crop_path])
    for (x1, y1, x2, y2) in boxes_den:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for idx, (x1, y1, x2, y2) in enumerate(boxes_ap, start=1):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_rgb, str(idx), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    marked_path = os.path.join(RESULTS_DIR, f"marked_{img_name}")
    cv2.imwrite(marked_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

with pd.ExcelWriter(os.path.join(RESULTS_DIR, "stomata_results.xlsx")) as writer:
    df_den = pd.DataFrame(density_rows,
                          columns=["Image", "Stomata Count", "Area (mm²)", "Density (stomata/mm²)"])
    df_cr  = pd.DataFrame(crop_rows,
                          columns=["Image", "Stoma Number", "Stomata Filepath"])
    df_den.to_excel(writer, sheet_name="Stomata_density", index=False)
    df_cr.to_excel(writer, sheet_name="Stomata_aperture",  index=False)

print("▶ Phase 1 completed: detection and density.")
print("  - Marked images in ./Results")
print("  - Cropped images in ./Results/Stomata")
print("  - Excel with Stomata_density and Stomata_crops")

### STOMATA APERTURE ###

CROPS_DIR    = "./Results/Stomata"
RESULTS_FILE = "./Results/stomata_results.xlsx"
MODEL_PATH   = "stomata_model.pt"
CLS_THRESH   = 0.5  ## Classification (NC vs aperture) prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StomataNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),])

results = []
for fname in sorted(os.listdir(CROPS_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    crop_path = os.path.join(CROPS_DIR, fname)
    try:
        img = Image.open(crop_path).convert("RGB")
    except Exception as e:
        print(f"Error with {fname}: {e}")
        continue
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_logit, reg_out = model(tensor)
        prob = torch.sigmoid(cls_logit).item()
        if prob < CLS_THRESH:
            pred_val = None
        else:
            pred_val = float(reg_out.item())
    pred_cls = 0 if pred_val is None else 1
    results.append({
        "filename": fname,
        "pred": pred_val,
        "pred_cls": pred_cls
    })

df_results = pd.DataFrame(results)
if os.path.exists(RESULTS_FILE):
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_results.to_excel(writer, sheet_name="Stomata_aperture", index=False)
else:
    with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Stomata_aperture", index=False)

print("▶ Phase 2 completed: aperture prediction.")
print(f"Results saved as 'Stomata_aperture' in {RESULTS_FILE}")