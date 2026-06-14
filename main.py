import sys
import os
sys.path.append('src')

from disease_detector import train

if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "PlantVillage", "PlantVillage")
    model, le = train(DATA_DIR)
    print("Training complete!")