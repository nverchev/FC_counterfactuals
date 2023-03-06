import os
import pandas as pd
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from options import SVCParser as Parser
from src.dataset import get_loaders
from src.loss_and_metrics import ClassMetrics
args = Parser().parse_args()

final = args.exp[:5] == 'final'
args.final = final

train_loader, val_loader, test_loader = get_loaders(**vars(args))
train_data = train_loader.dataset.matrices
train_labels = train_loader.dataset.labels
train_metadata = train_loader.dataset.metadata
model_path = os.path.join(args.data_dir, 'models', '_'.join(filter(bool, ['svc', args.exp])) + '.joblib')
model = SVC(kernel=args.kernel, C=args.C, gamma='auto', random_state=args.seed, probability=True)
if not args.eval:
    print('Fitting the model ...')
    model.fit(train_data, train_labels)
    dump(model, model_path)
else:
    model = load(model_path)
    print('Loaded model in ', model_path)
if not final:
    scores = cross_val_score(model, train_data, train_labels, cv=5)
    for i, score in enumerate(scores):
        print(f'Section {i} accuracy: {score:.2f}')
else:
    test_data = test_loader.dataset.matrices
    test_labels = test_loader.dataset.labels
    test_metadata = test_loader.dataset.metadata
    pred = model.predict(test_data)
    ClassMetrics()(test_labels, pred)
    train_metadata['svc_prob'] = model.predict_proba(train_data)[:, 1]
    test_metadata['svc_prob'] = model.predict_proba(test_data)[:, 1]
    metadata = pd.concat([train_metadata, test_metadata])
    metadata_pth = os.path.join(args.data_dir, 'metadata.csv')
    orig_metadata = pd.read_csv(metadata_pth)['file']
    a = pd.merge(orig_metadata, metadata, on='file', validate='1:1').to_csv(metadata_pth, index=False)
    print(pd.read_csv(metadata_pth).head())
