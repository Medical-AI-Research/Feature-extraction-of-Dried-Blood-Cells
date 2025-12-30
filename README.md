# Explainable Hybrid Deep Feature Learning for Subject Identification from Dried Blood Droplet Images

Research-only project implementing a hybrid deep–classical ML framework to identify subject-specific visual signatures from dried blood droplet (DBD) images using explainable AI.

Dataset: Figshare (Hamadeh et al., 2019) | 1710 images | 30 healthy subjects  
Acquisition: controlled drying on glass slides, CASIO EX-Z1000 HDR macro, ~100 px/mm, single droplet per image.

Environment:
Python >= 3.9  
pip install torch torchvision timm scikit-learn shap grad-cam matplotlib numpy

Data Layout:
DBD_Dataset/subject_01 … subject_30 (images)

Preprocessing:
- Resize 224×224
- ImageNet normalization
- No histogram equalization
- No augmentation
- 5-fold stratified cross-validation

Feature Extraction (Frozen Backbone):
import timm  
backbone = timm.create_model("densenet201", pretrained=True, num_classes=0)  
Global Average Pooling → 1920-D feature vector

Classifier:
from sklearn.ensemble import RandomForestClassifier  
clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", bootstrap=True, random_state=42)

Training:
- Extract deep features per image
- Train RF on features
- Evaluate per fold, aggregate results

Evaluation:
- Mean accuracy (5-fold CV)
- Aggregated confusion matrix  
Performance: DenseNet201 + RF = 82.5% (outperforms softmax and MobileNet variants)

Explainability:
Grad-CAM (spatial relevance): peripheral ring, radial cracks, internal texture  
from grad_cam import GradCAM  
cam = GradCAM(model=backbone, target_layer="features.denseblock4")

SHAP (feature importance): compact subset of deep features dominates decisions  
import shap  
explainer = shap.TreeExplainer(clf)

Outputs:
outputs/confusion_matrix.png  
outputs/gradcam_visualizations/  
outputs/shap_feature_importance.png  
outputs/metrics.json

Findings:
- DBD morphology encodes stable subject-level information
- Hybrid deep features + RF improves robustness
- XAI aligns with known physical drying phenomena

Limitations:
- Controlled acquisition only
- Healthy subjects only
- Identification, not biometric authentication

Future Work:
- Larger/diverse cohorts
- Pathological samples
- Attention mechanisms
- Integration with physical/biochemical features

License: Apache License 2.0  
Disclaimer: Research and academic use only; not for clinical or forensic decisions.  
Author: Kethan — Biomedical AI / Explainable Medical Imaging
