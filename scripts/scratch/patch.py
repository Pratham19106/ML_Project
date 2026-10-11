import re

with open('train_freq_cbm.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('from src.dataset import GTSRBDataset', 'from src.dataset import GTSRBDataset\nfrom src.augmentations.frequency import FrequencyAugDataset')

# Replace train_dataset setup using regex to handle whitespace safely
c = re.sub(
    r'train_dataset = GTSRBDataset\(\s*manifest_path=data_root / "master_v1/metadata/train\.csv",\s*data_root=data_root,\s*is_training=True,\s*\)',
    '''train_dataset = FrequencyAugDataset(
        manifest_path=data_root / "master_v1/metadata/train.csv",
        data_root=data_root,
        is_training=True,
        mode="dropout",
        radius=14,
        severity=3.0,
    )''',
    c
)

c = c.replace('checkpoints/cbm_resnet18', 'checkpoints/freq_cbm_resnet18')
c = c.replace('best_cbm_resnet18.pt', 'best_freq_cbm_resnet18.pt')
c = c.replace('Experiment 4', 'Experiment 6')
c = c.replace('Concept Bottleneck ResNet-18', 'Frequency-Aware Concept Bottleneck Model')
c = c.replace('Starting CBM Training...', 'Starting Frequency-Aware CBM Training...')

with open('train_freq_cbm.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Patched train_freq_cbm.py successfully!')
