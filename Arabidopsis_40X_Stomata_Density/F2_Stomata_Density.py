## pip install pandas openpyxl
## pip install ultralytics
import os
import pandas as pd
import argparse
from ultralytics import YOLO

def main(threshold):
    model = YOLO("./runs/detect/train/weights/best.pt")
    input_folder = "./Input"
    output_folder = "./Results/Images"
    os.makedirs(output_folder, exist_ok=True)
    input_images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    results_list = []
    for image in input_images:
        full_input = os.path.join(input_folder, image)
        results = model(full_input, conf=threshold, save=True, project="./Results", name="Images", exist_ok=True)
        count_stomata = 0
        for result in results:
            count_stomata += len(result.boxes)
        width, height = result.orig_shape[1], result.orig_shape[0]
        if width == 1920 and height == 1080:
            area_mm2 = (321 * 180) / 1000  # Area (mm²)
        elif width == 1600 and height == 1200:
            area_mm2 = (268 * 201) / 1000  # Area (mm²)
        else:
            print(f"Unknown size for image {image}. Density will not be calculated.")
            area_mm2 = None
        if area_mm2:
            density = count_stomata / area_mm2
            results_list.append([image, count_stomata, area_mm2, density])
    df = pd.DataFrame(results_list, columns=["Image", "Stomata Count", "Area (mm²)", "Density (stomata/mm²)"])
    os.makedirs("./Results", exist_ok=True)
    df.to_excel("./Results/results_density.xlsx", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("threshold", nargs="?", type=float, default=None, help="Confidence threshold (default=0.18)")
    args = parser.parse_args()
    if args.threshold is None:
        try:
            args.threshold = float(input("Insert threshold (default 0.18): ") or 0.18)
        except ValueError:
            print("Invalid value. Using threshold = 0.18")
            args.threshold = 0.18
    main(args.threshold)
