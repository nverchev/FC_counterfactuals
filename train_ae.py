from options import AEParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)

    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint)
            trainer.save()
            #trainer.test('val')
    else:
        trainer.test(partition='val', save_outputs=True)


if __name__ == '__main__':
    main()
