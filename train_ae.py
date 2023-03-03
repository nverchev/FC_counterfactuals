from options import AEParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)
    final = args.exp[:5] == 'final'
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint)
            trainer.save()
            if not final:
                trainer.test('val')
    else:
        trainer.test('test' if final else 'val')
    for i in args.ind:
        trainer.viz_sample(i)

if __name__ == '__main__':
    main()
