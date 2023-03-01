import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from options import SVCParser as Parser
from src.dataset import get_loaders

args = Parser().parse_args()

train_loader, val_loader, test_loader = get_loaders(**vars(args))
train_data = train_loader.dataset.matrices
train_labels = train_loader.dataset.labels

model = SVC(kernel=args.kernel, C=args.C, gamma='auto', random_state=args.seed, probability=True)
print('Fitting the model ...')
model.fit(train_data, train_labels)

pickle.dump(model, open(os.path.join(args.data_dir, args.exp), 'wb'))

scores = cross_val_score(model, train_data, train_labels, cv=5)
print(scores)