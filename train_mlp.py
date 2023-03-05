import os
import json
import torch
import pandas as pd
from options import MLPParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    exp = args.exp
    if args.pretrain:
        args.exp = 'pretrain_' + args.exp
    trainer = get_trainer(args)
    final = args.exp[:5] == 'final'
    if args.pretrain and args.load == -1:
        ae_path_dir = os.path.join(args.data_dir, 'models', '_'.join(filter(bool, ['ae',  exp])))
        final_epoch = json.load(open(os.path.join(ae_path_dir, 'settings.json')))['training_epochs']
        ae_path = os.path.join(ae_path_dir, f'model_epoch{final_epoch}.pt')
        state = torch.load(ae_path, map_location=torch.device(trainer.device))
        trainer.model.load_state_dict(state, strict=False)
        print('Pretrained weights in ' + ae_path + ' loaded.')
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint, val_after_train=not final)
            trainer.save()
            if not final:
                trainer.test('val')
    if final:
        prob_name = 'ae_prob' if args.pretrain else 'mlp_prob'
        train_metadata = trainer.update_metadata(partition='train', prob_name=prob_name)
        test_metadata = trainer.update_metadata(partition='test', prob_name=prob_name)
        metadata = pd.concat([train_metadata, test_metadata])
        metadata_pth = os.path.join(args.data_dir, 'metadata.csv')
        orig_metadata = pd.read_csv(metadata_pth)['file']
        pd.merge(orig_metadata, metadata, on='file', validate='1:1').to_csv(metadata_pth, index=False)
    else:
        trainer.test(partition='val', save_outputs=True)


if __name__ == '__main__':
    main()
