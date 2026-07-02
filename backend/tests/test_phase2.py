import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dataset import SudokuDigitDataset
from model import DigitRecognizerCNN

def evaluate_model():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/datasets'))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/digit_recognizer.pth'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/output/phase2'))
    os.makedirs(output_dir, exist_ok=True)
    
    transform_test = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading test dataset...")
    test_dataset = SudokuDigitDataset(data_dir, split='test', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DigitRecognizerCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title('Confusion Matrix - Digit Recognition')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print(f"Evaluation complete. Results saved to {output_dir}")
    print("\nClassification Report:\n")
    print(report)

if __name__ == '__main__':
    evaluate_model()
