import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from F4_ASAP_Model import StomataNet

### STOMATA APERTURE ###

INPUT_DIR     = "./Input"
RESULTS_DIR   = "./Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE  = os.path.join(RESULTS_DIR, "stomata_aperture_from_crops.xlsx")
MODEL_PATH    = "stomata_model.pt"
CLS_THRESH    = 0.5   ## aperture threshold

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
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    crop_path = os.path.join(INPUT_DIR, fname)
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
        "pred_cls": pred_cls})

### SAVE RESULTS ###

df_results = pd.DataFrame(results)
df_results.to_excel(RESULTS_FILE, sheet_name="Stomata_aperture", index=False)
print("▶ Aperture prediction completed.")
print(f"Results saved in {RESULTS_FILE}")