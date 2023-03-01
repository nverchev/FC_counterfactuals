from options import AEParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint)
            trainer.save()
            if args.exp[:5] != 'final':
                pass
                #trainer.test('val')

    if args.exp[:5] == 'final':
        trainer.test(partition='train', save_outputs=True)
        trainer.test(partition='test', save_outputs=True)


    else:
        trainer.test('val')


if __name__ == '__main__':
    main()
