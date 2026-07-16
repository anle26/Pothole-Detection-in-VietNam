"""Find which source folders the UNKNOWN images belong to."""
import os

base = r"D:\Research\Yolo_Pothole_detection\Pothole-Detection-in-VietNam\data\VietNamDataset"

unknowns = [
    '1217135530503348', '1228642296139397', '1229995555988658', '1241758681476667',
    '1434194691529387', '1442024617298696', '1464944235336441', '1470709297996270',
    '1491652285926735', '1620619455789572', '1943654982927086', '787664853903845',
    '930804512874124', '932698522466670', '958933323156298', '1249599540065346',
    '1277903437622494', '1404578194242137', '1483238870097287', '25938274625800361'
]

folders = [item for item in os.listdir(base) 
           if os.path.isdir(os.path.join(base, item)) 
           and item not in ('train', 'valid', 'test', 'full_dataset')]

for uid in unknowns:
    found = False
    for item in folders:
        files = os.listdir(os.path.join(base, item))
        for f in files:
            if f.startswith(uid):
                print(f"{uid} -> {item}")
                found = True
                break
        if found:
            break
    if not found:
        print(f"{uid} -> NOT FOUND IN ANY FOLDER")
