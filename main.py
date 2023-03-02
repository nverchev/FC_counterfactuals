from options import MainParser as Parser
from src.trainers import get_trainer


def main():
    args = Parser().parse_args()
    trainer = get_trainer(args)
    final = args.exp[:5] == 'final'
    if args.n_gen:
        trainer.generate_samples(args.n_gen)
    if not args.eval:
        while args.epochs > trainer.epoch:
            trainer.train(args.checkpoint)
            trainer.save()
            if not final:
                trainer.test('val')
    trainer.test('test' if final else 'val')
    for i in args.ind:
        trainer.viz_sample(i)
        trainer.generate_counterfactuals(i)


if __name__ == '__main__':
    main()
