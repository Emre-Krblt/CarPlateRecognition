import os
import cv2
import easyocr
import re
import numpy as np
import csv

image_folder = "BIM472_Project1_Images"
output_folder = "outputs"
csv_path = "detected_plates.csv"
os.makedirs(output_folder, exist_ok=True)

reader = easyocr.Reader(['en'])

def is_valid_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    return 4 <= len(text) <= 10 and re.match(r'^[A-Z0-9]+$', text)

# CSV dosyasını başlıkla oluştur
with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Image", "DetectedPlate", "Confidence"])

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            print(f"\n🖼️ Image: {filename}")

            results = reader.readtext(img)
            found = False

            for (bbox, text, prob) in results:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                print(f"  📝 Detected: {text} → Cleaned: {cleaned} (Confidence: {prob:.2f})")

                if is_valid_plate(cleaned):
                    pts = [tuple(map(int, point)) for point in bbox]
                    cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, cleaned, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # 🔽 CSV’ye yaz
                    csvwriter.writerow([filename, cleaned, f"{prob:.2f}"])

                    found = True
                    print(f"  ✅ Plate Detected: {cleaned}")
                    break  # ilk tespitte dur

            if not found:
                print("❌ No valid plate found.")

            # 📸 Görseli kaydet
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)

print("\n✅ Tüm işlemler tamamlandı. Detaylı sonuçlar detected_plates.csv içinde.")
