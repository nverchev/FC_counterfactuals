import os
import json
import torch
from options import MLPParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)
    if args.pretrain:
        ae_path_dir = os.path.join(args.data_dir, 'models', 'ae' + trainer.exp_name[4:])
        final_epoch = json.load(open(os.path.join(ae_path_dir, 'settings.json')))['training_epochs']
        ae_path = os.path.join(ae_path_dir, f'model_epoch{final_epoch}.pt')
        state = torch.load(ae_path, map_location=torch.device(trainer.device))
        trainer.model.load_state_dict(state, strict=False)
        print('Pretrained weights in ' + ae_path + ' loaded.')
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint)
            trainer.save()
            #trainer.test('val')
    else:
        trainer.test(partition='val', save_outputs=True)


if __name__ == '__main__':
    main()
