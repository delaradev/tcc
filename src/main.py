"""
================================================================================
CPIC BRAZIL - MAIN ENTRY POINT
================================================================================

Main entry point for the Center Pivot Irrigation Cropland (CPIC) mapping pipeline.

Provides unified CLI interface for:
    - Training: Train U-Net model on CPIC dataset
    - Prediction: Run inference on new areas
    - Export: Export trained model for deployment

Based on methodology from:
Liu et al. (2023) - ISPRS Journal of Photogrammetry and Remote Sensing

Author: Emanuel de Lara Ruas
Version: 1.0.0
================================================================================
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress TensorFlow warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cpic_pipeline.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------

def setup_paths() -> None:
    """
    Configure Python path to include src directory for imports

    This allows running the script from any location while maintaining
    consistent import resolution.
    """
    project_root = Path(__file__).resolve().parent
    src_path = project_root / 'src'

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.debug(f"Added {src_path} to Python path")


# -----------------------------------------------------------------------------
# MODULE IMPORTS (lazy loading for better performance)
# -----------------------------------------------------------------------------

def get_trainer(config_path: str):
    """Lazy import Trainer to avoid loading TensorFlow when not needed"""
    try:
        from src.training.train import Trainer
        return Trainer(config_path)
    except ImportError as e:
        logger.error(f"Failed to import Trainer: {e}")
        logger.error("Make sure src/training/train.py exists")
        sys.exit(1)


def get_predictor(model_path: str, config_path: Optional[str] = None):
    """Lazy import Predictor for inference mode"""
    try:
        from src.inference.predict import CPICPredictor
        return CPICPredictor(model_path, config_path)
    except ImportError as e:
        logger.error(f"Failed to import CPICPredictor: {e}")
        logger.error("Inference module not yet implemented or missing")
        sys.exit(1)


def get_exporter(model_path: str, output_path: str):
    """Lazy import Exporter for model export"""
    try:
        from src.export.model_exporter import CPICExporter
        return CPICExporter(model_path, output_path)
    except ImportError as e:
        logger.error(f"Failed to import CPICExporter: {e}")
        logger.error("Export module not yet implemented or missing")
        sys.exit(1)


# -----------------------------------------------------------------------------
# CLI ARGUMENT PARSING
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments with comprehensive help

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='CPIC Brazil - Center Pivot Irrigation Cropland Mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train new model with default config
  python main.py --mode train --config config/config.yaml

  # Train with custom parameters
  python main.py --mode train --config config/config.yaml --epochs 100 --batch_size 2

  # Run prediction on new area
  python main.py --mode predict --model runs/best_model.keras --input data/new_area.tif

  # Export model for deployment
  python main.py --mode export --model runs/best_model.keras --output models/exported/

For more information, see documentation: https://github.com/delaradev/
        """
    )

    # Common arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'export', 'validate'],
        default='train',
        help='Execution mode (default: train)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Training mode arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of training epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )

    # Prediction mode arguments
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (.keras format)'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input image or directory for prediction'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for predictions'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for prediction (default: 0.5)'
    )

    # Validation mode arguments
    parser.add_argument(
        '--test_split',
        type=str,
        default='valid',
        help='Dataset split to use for validation (default: valid)'
    )

    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help='Device to use for computation (default: auto)'
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# EXECUTION MODES
# -----------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """
    Execute training mode

    Args:
        args: Parsed command line arguments
    """
    logger.info("=" * 70)
    logger.info("CPIC BRAZIL - TRAINING MODE")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please ensure config/config.yaml exists")
        sys.exit(1)

    # Initialize trainer
    trainer = get_trainer(args.config)

    # Override parameters if provided
    if args.epochs is not None:
        trainer.config['training']['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")

    if args.batch_size is not None:
        trainer.config['training']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")

    if args.resume:
        trainer.resume_from = args.resume
        logger.info(f"Resuming from checkpoint: {args.resume}")

    # Configure device
    if args.device == 'cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("Forcing CPU usage")

    # Run training
    try:
        trainer.run()
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_predict(args: argparse.Namespace) -> None:
    """
    Execute prediction mode

    Args:
        args: Parsed command line arguments
    """
    logger.info("=" * 70)
    logger.info("CPIC BRAZIL - PREDICTION MODE")
    logger.info("=" * 70)

    # Validate required arguments
    if args.model is None:
        logger.error("--model is required for prediction mode")
        sys.exit(1)

    if args.input is None:
        logger.error("--input is required for prediction mode")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Initialize predictor
    predictor = get_predictor(str(model_path), args.config)

    # Set threshold
    predictor.threshold = args.threshold

    # Run prediction
    try:
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            result = predictor.predict_file(input_path, args.output)
        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            result = predictor.predict_directory(input_path, args.output)
        else:
            logger.error(f"Invalid input path: {input_path}")
            sys.exit(1)

        logger.info(f"✅ Prediction completed. Output: {result}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_export(args: argparse.Namespace) -> None:
    """
    Execute export mode

    Args:
        args: Parsed command line arguments
    """
    logger.info("=" * 70)
    logger.info("CPIC BRAZIL - EXPORT MODE")
    logger.info("=" * 70)

    # Validate required arguments
    if args.model is None:
        logger.error("--model is required for export mode")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    output_path = args.output or str(model_path.parent / 'exported')

    # Initialize exporter
    exporter = get_exporter(str(model_path), output_path)

    # Run export
    try:
        exported_paths = exporter.export()
        logger.info("Model exported successfully")
        for path in exported_paths:
            logger.info(f"   - {path}")
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def run_validate(args: argparse.Namespace) -> None:
    """
    Execute validation mode

    Args:
        args: Parsed command line arguments
    """
    logger.info("=" * 70)
    logger.info("CPIC BRAZIL - VALIDATION MODE")
    logger.info("=" * 70)

    # Validate required arguments
    if args.model is None:
        logger.error("--model is required for validation mode")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Initialize predictor for validation
    predictor = get_predictor(str(model_path), args.config)

    # Run validation
    try:
        results = predictor.validate(split=args.test_split)

        logger.info("\nVALIDATION RESULTS:")
        logger.info(f"  IoU:        {results.get('iou', 0):.4f}")
        logger.info(f"  Dice:       {results.get('dice', 0):.4f}")
        logger.info(f"  Precision:  {results.get('precision', 0):.4f}")
        logger.info(f"  Recall:     {results.get('recall', 0):.4f}")
        logger.info(f"  F1-score:   {results.get('f1', 0):.4f}")

        if 'confusion_matrix' in results:
            logger.info(f"  Confusion Matrix: {results['confusion_matrix']}")

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for CPIC Brazil pipeline

    Parses arguments and dispatches to appropriate execution mode
    """
    # Setup paths for imports
    setup_paths()

    # Parse command line arguments
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Display banner
    logger.info("=" * 70)
    logger.info("CPIC Brazil - Center Pivot Irrigation Cropland Mapping")
    logger.info("   Based on Liu et al. (2023) - ISPRS Journal")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode.upper()}")

    # Dispatch to appropriate mode
    mode_handlers = {
        'train': run_train,
        'predict': run_predict,
        'export': run_export,
        'validate': run_validate
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
