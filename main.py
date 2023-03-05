from options import MainParser as Parser
from src.trainers import get_trainer
import os
import json
import joblib


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)
    final = args.exp[:5] == 'final'
    if args.gen:
        trainer.generate_samples(args.gen)
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint, val_after_train=not final)
            trainer.save()
            if not final:
                trainer.test('val')
    trainer.test('test' if final else 'val', save_outputs=args.ind)
    for i in args.ind:
        trainer.viz_sample(i)
        trainer.generate_counterfactuals(i)

    model_path = os.path.join(args.data_dir, 'models')
    if args.cond == 'svc':
        pass
    svc_model = joblib.load(os.path.join(model_path, '_'.join(filter(bool, ['svc', args.exp])) + '.joblib'))
    model = lambda x: svc_model.predict_proba(x)[:, 1]
    # elif args.cond == 'svc':
    #     joblib.load()
    #     svc_model = joblib.load(os.path.join(model_path, '_'.join(filter(bool, ['svc', args.exp])) + '.joblib'))
    #     model = lambda x: svc_model.predict_proba(x)[:, 1]
    # model_path_dir = os.path.join(args.data_dir, 'models', '_'.join(filter(bool, [args.cond, args.exp])))
    # final_epoch = json.load(open(os.path.join(ae_path_dir, 'settings.json')))['training_epochs']
    # ae_path = os.path.join(ae_path_dir, f'model_epoch{final_epoch}.pt')
    # state = torch.load(ae_path, map_location=torch.device(trainer.device))
    # trainer.model.load_state_dict(state, strict=False)
    # print('Pretrained weights in ' + ae_path + ' loaded.')
    trainer.evaluate_counterfactuals(model=model)


if __name__ == '__main__':
    main()
