
from common.dental_data import preprocess_data_and_save

        
data_path = "./dental_data/Data/processed_data.json"

data = preprocess_data_and_save(data_path)

features = data["features"]
labels = data["labels"]

total = len(features)
features = len(features[0])
print(len(labels))
positive = sum(labels)
negative = total - positive
print(f"Total samples: {total}, Positive samples: {positive}, Negative samples: {negative}, ")
print(f"Features per sample: {features}")

print("Data processing pipeline completed successfully.")
