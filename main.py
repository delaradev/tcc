"""Ponto de entrada principal do projeto"""
import argparse
from pathlib import Path
from training.train import Trainer


def main():
    parser = argparse.ArgumentParser(
        description='Treinamento de pivôs centrais')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Caminho do arquivo de configuração'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'export'],
        default='train',
        help='Modo de execução'
    )
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer(args.config)
        trainer.run()

    elif args.mode == 'predict':
        # TODO: Implementar inferência
        print("Modo predição (em desenvolvimento)")

    elif args.mode == 'export':
        # TODO: Exportar para GEE
        print("Modo exportação (em desenvolvimento)")


if __name__ == '__main__':
    main()
