import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from options import parse_svc_args
from src.dataset import get_loaders

args = parse_svc_args().parse_args()
exp_name = 'svc' + '_' * bool(args.exp) + args.exp
dir_path = args.dir_path
data_dir = os.path.join(dir_path, 'abide_fc_dataset')
kernel = args.kernel
C = args.C
data_loader_settings = dict(
  data_dir=data_dir,
  jitter=True,
  random_crop=True,
  batch_size=1,
)
train_loader, val_loader, test_loader = get_loaders(**data_loader_settings)
train_data = train_loader.dataset.matrices
train_labels = train_loader.dataset.labels

model = SVC(kernel=kernel, C=C, gamma='auto', random_state=2023, probability=True)
model.fit(train_data, train_labels)
pickle.dump(model, open(os.path.join(data_dir, exp_name), 'wb'))

scores = cross_val_score(model, train_data, train_labels, cv=5)
print(scores)