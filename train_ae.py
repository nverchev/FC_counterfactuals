from options import AEParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    final = args.exp[:5] == 'final'
    trainer = get_trainer(args, final=final)
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint, val_after_train=not final)
            trainer.save()
            if not final:
                trainer.test('val')
    trainer.test('test' if final else 'val', save_outputs=args.ind)
    for i in args.ind:
        trainer.viz_sample(i)


if __name__ == '__main__':
    main()
