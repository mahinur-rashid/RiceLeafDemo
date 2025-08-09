#!/usr/bin/env python3
"""
🌾 ULTRA-OPTIMIZED Rice Disease Detection Model
Target: 95%+ Accuracy with Stable Training
Features: EfficientNet-V2, Advanced Augmentation, Stable Learning Rate, Multi-Scale Features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import pandas as pd
from collections import Counter
import random
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class AdvancedRiceDataset(Dataset):
    """Advanced dataset with smart augmentation and balanced sampling"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image and label 0 as fallback
            if self.transform:
                fallback_image = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
                image = self.transform(fallback_image)
            else:
                image = torch.zeros(3, 320, 320)
            return image, 0

def get_advanced_transforms():
    """Get enhanced transforms with better regularization for rice leaf analysis"""
    
    # Enhanced training transforms with CutMix-style augmentation
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomResizedCrop(320, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value='random'
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Validation/Test transforms - clean and consistent
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform

class UltraOptimizedRiceClassifier(nn.Module):
    """Ultra-optimized EfficientNet-V2 Large model with enhanced regularization"""
    
    def __init__(self, num_classes=8, dropout_rate=0.5):
        super(UltraOptimizedRiceClassifier, self).__init__()
        
        # Load EfficientNet-V2 Large (most powerful variant)
        self.backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        
        # Get the number of features from the classifier
        backbone_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Advanced multi-scale feature extraction with feature dropout
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Dropout(0.2)  # Feature-level dropout
        )
        
        # Enhanced attention mechanism with dropout
        self.attention = nn.MultiheadAttention(
            embed_dim=backbone_features,
            num_heads=8,  # Reduced heads to prevent overfitting
            dropout=0.2,
            batch_first=True
        )
        
        # Layer normalization for attention
        self.attention_norm = nn.LayerNorm(backbone_features)
        
        # Progressive regularization classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9),  # Increased dropout
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Progressive dropout
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            # Additional regularization layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features with dropout
        features = self.feature_extractor(x)  # [batch_size, backbone_features]
        
        # Apply attention with normalization
        features_att = features.unsqueeze(1)  # [batch_size, 1, backbone_features]
        attended_features, _ = self.attention(features_att, features_att, features_att)
        attended_features = attended_features.squeeze(1)  # [batch_size, backbone_features]
        
        # Apply layer normalization
        attended_features = self.attention_norm(attended_features)
        
        # Residual connection with reduced weight
        combined_features = features + 0.3 * attended_features  # Reduced attention weight
        
        # Classify
        output = self.classifier(combined_features)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def load_optimized_dataset(data_dir, test_size=0.15, val_size=0.15):
    """Load dataset with intelligent oversampling and class balancing"""
    
    print("\n🔍 Loading and analyzing dataset...")
    
    image_paths = []
    labels = []
    class_names = []
    
    # Collect all images and labels
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            class_images = []
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    class_images.append(img_path)
            
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            print(f"  {class_name}: {len(class_images)} images")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Intelligent oversampling strategy
    print("\n🎯 Applying intelligent oversampling...")
    class_counts = Counter(labels)
    max_samples = max(class_counts.values())
    target_samples = min(max_samples, 800)  # Cap at 800 for computational efficiency
    
    balanced_paths = []
    balanced_labels = []
    
    for class_idx in range(len(class_names)):
        class_mask = labels == class_idx
        class_paths = image_paths[class_mask]
        current_count = len(class_paths)
        
        if current_count < target_samples:
            # Oversample this class
            needed_samples = target_samples - current_count
            oversample_indices = np.random.choice(
                len(class_paths), 
                needed_samples, 
                replace=True
            )
            oversample_paths = class_paths[oversample_indices]
            
            balanced_paths.extend(class_paths)
            balanced_paths.extend(oversample_paths)
            balanced_labels.extend([class_idx] * target_samples)
            
            print(f"  Oversampled {class_names[class_idx]}: {current_count} -> {target_samples}")
        else:
            # Keep original samples
            balanced_paths.extend(class_paths)
            balanced_labels.extend([class_idx] * current_count)
            print(f"  Kept {class_names[class_idx]}: {current_count} images")
    
    # Convert back to numpy arrays
    balanced_paths = np.array(balanced_paths)
    balanced_labels = np.array(balanced_labels)
    
    # Split dataset
    print(f"\n📊 Final dataset size: {len(balanced_paths)} images")
    
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        balanced_paths, balanced_labels, 
        test_size=test_size, 
        stratify=balanced_labels, 
        random_state=42
    )
    
    # Second split: separate train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size/(1-test_size),  # Adjust for already split test set
        stratify=train_val_labels,
        random_state=42
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names

def create_optimized_data_loaders(train_data, val_data, batch_size=16):
    """Create optimized data loaders with weighted sampling"""
    
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    
    train_transform, val_transform = get_advanced_transforms()
    
    # Create datasets
    train_dataset = AdvancedRiceDataset(train_paths, train_labels, train_transform, is_training=True)
    val_dataset = AdvancedRiceDataset(val_paths, val_labels, val_transform, is_training=False)
    
    # Create weighted sampler for training
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def train_optimized_model(model, train_loader, val_loader, num_epochs=120, initial_lr=8e-5):
    """Train model with enhanced regularization and stable training strategy"""
    
    print(f"\n🚀 Starting enhanced regularized training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate class weights for focal loss
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    class_counts = Counter(all_labels)
    num_classes = len(class_counts)
    class_weights = torch.tensor([
        len(all_labels) / (num_classes * class_counts[i]) 
        for i in range(num_classes)
    ], dtype=torch.float32).to(device)
    
    # Enhanced loss function with label smoothing
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Enhanced optimizer with better weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.02,  # Increased weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Stable learning rate scheduler without restarts
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=True
    )
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience = 20  # Reduced patience
    patience_counter = 0
    
    # Mixup parameters for additional regularization
    mixup_alpha = 0.2
    
    for epoch in range(num_epochs):
        # Training phase with enhanced regularization
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Apply mixup augmentation randomly
            if np.random.random() < 0.3:  # 30% chance of mixup
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(device)
                
                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                
                optimizer.zero_grad()
                outputs = model(mixed_images)
                
                # Mixup loss
                loss_a = criterion(outputs, labels_a)
                loss_b = criterion(outputs, labels_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                
                # Add label smoothing component
                smooth_loss = label_smoothing_loss(outputs, labels_a)
                loss = 0.9 * loss + 0.1 * smooth_loss
                
            else:
                # Regular training
                optimizer.zero_grad()
                outputs = model(images)
                
                # Combined loss: Focal + Label Smoothing
                focal_loss = criterion(outputs, labels)
                smooth_loss = label_smoothing_loss(outputs, labels)
                loss = 0.8 * focal_loss + 0.2 * smooth_loss
            
            loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
            
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Update learning rate based on validation accuracy
        scheduler.step(epoch_val_acc)
        
        # Save metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 60)
        
        # Early stopping and model saving
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            print(f"🎯 New best validation accuracy: {best_val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }, 'best_rice_model_regularized.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏰ Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Check if target accuracy reached
        if epoch_val_acc >= 95.0:
            print(f"\n🎉 TARGET ACHIEVED! Validation accuracy: {epoch_val_acc:.2f}% >= 95%")
            break
    
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def test_time_augmentation(model, image, transform, num_augmentations=10):
    """Apply test time augmentation for better predictions"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_augmentations):
            augmented_image = transform(image).unsqueeze(0).to(device)
            output = model(augmented_image)
            predictions.append(torch.softmax(output, dim=1))
    
    # Average predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction

def evaluate_model(model, test_data, class_names, batch_size=16):
    """Comprehensive model evaluation with TTA"""
    
    test_paths, test_labels = test_data
    _, val_transform = get_advanced_transforms()
    
    test_dataset = AdvancedRiceDataset(test_paths, test_labels, val_transform, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("\n🧪 Evaluating model with Test Time Augmentation...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 Test Results:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed classification report
    print("\n📈 Detailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Optimized Rice Disease Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def plot_training_history(history):
    """Plot training history"""
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss', color='blue')
    ax1.plot(history['val_losses'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='red')
    ax2.axhline(y=95, color='green', linestyle='--', label='Target 95%')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    
    print("🌾 ULTRA-OPTIMIZED Rice Disease Detection Model")
    print("=" * 60)
    print("Target: 95%+ Accuracy with Enhanced Regularization")
    print("Features: EfficientNet-V2-L, Mixup, Label Smoothing, Stable LR")
    print("=" * 60)
    
    # Enhanced configuration for better regularization
    data_dir = r'E:\riceleafML\dataset'
    batch_size = 10  # Reduced batch size for better generalization
    num_epochs = 120  # Reduced epochs with better regularization
    learning_rate = 8e-5  # Slightly lower learning rate
    
    # Load dataset
    train_data, val_data, test_data, class_names = load_optimized_dataset(data_dir)
    
    # Create data loaders
    train_loader, val_loader = create_optimized_data_loaders(train_data, val_data, batch_size)
    
    # Create model
    print(f"\n🤖 Creating Ultra-Optimized Model...")
    model = UltraOptimizedRiceClassifier(num_classes=len(class_names)).to(device)
    
    print(f"Model: EfficientNet-V2-L with Multi-Head Attention")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    trained_model, history = train_optimized_model(
        model, train_loader, val_loader, num_epochs, learning_rate
    )
    
    # Load best model for evaluation
    checkpoint = torch.load('best_rice_model_regularized.pth')
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_accuracy = evaluate_model(trained_model, test_data, class_names, batch_size)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    if history['best_val_acc'] >= 95.0:
        print("🎉 SUCCESS: Target 95%+ accuracy achieved!")
    else:
        print("🔄 Continue training or adjust hyperparameters for 95%+ accuracy")
    
    return trained_model, class_names

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        trained_model, class_names = main()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n⏱️  Total training time: {training_time/3600:.2f} hours")
        print("🌾 Rice disease detection model training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
