import re

with open('train_freq_cbm.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('train_dataset = GTSRBDataset(train_manifest, args.data_root, is_training=True)',
    '''train_dataset = FrequencyAugDataset(
        manifest_path=train_manifest,
        data_root=args.data_root,
        is_training=True,
        mode="dropout",
        radius=14,
        severity=3.0,
    )''')

with open('train_freq_cbm.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Patched train_freq_cbm.py properly!')
