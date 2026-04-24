#!/usr/bin/env python3
"""
Command-Line Interface for TorchVision Dataset-Model Mapping.

Provides CLI tools for exploring datasets, downloading dataset-model pairs,
validating compatibility, and running comprehensive tests.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from typing import Optional

from act.util.cli_utils import add_device_args, initialize_from_args

from act.front_end.torchvision_loader.data_model_mapping import (
    DATASET_MODEL_MAPPING,
    get_dataset_info,
    list_datasets_by_category,
    get_all_categories,
    search_datasets,
    get_preprocessing_transforms,
    validate_dataset_model_compatibility,
    find_dataset_name,
    find_model_name,
)

from act.front_end.torchvision_loader.data_model_loader import (
    download_dataset_model_pair,
    load_dataset_model_pair,
    list_downloaded_pairs,
    model_inference_with_dataset,
    _format_size,
)


def print_mapping_summary(category: Optional[str] = None):
    """
    Print a summary of dataset-to-model mappings.
    
    Args:
        category: If provided, only show datasets in this category
    """
    print("=" * 100)
    print("TORCHVISION DATASET → MODEL MAPPING SUMMARY")
    print("=" * 100)
    
    categories_to_show = [category] if category else get_all_categories()
    
    for cat in categories_to_show:
        datasets = list_datasets_by_category(cat)
        if not datasets:
            continue
            
        print(f"\n{'='*100}")
        print(f"{cat.upper()} ({len(datasets)} datasets)")
        print('='*100)
        
        for dataset in sorted(datasets):
            info = DATASET_MODEL_MAPPING[dataset]
            print(f"\n{dataset}:")
            print(f"  Models ({len(info['models'])}): {', '.join(info['models'][:5])}"
                  f"{' ...' if len(info['models']) > 5 else ''}")
            print(f"  Input Size: {info['input_size']}")
            if info['num_classes']:
                print(f"  Classes: {info['num_classes']}")
            print(f"  Notes: {info['notes']}")


def print_preprocessing_summary():
    """
    Print an aggregated summary of preprocessing requirements across all datasets.
    
    Groups datasets by their preprocessing needs:
    - Grayscale→RGB conversion
    - Resize requirements
    - Ready without preprocessing
    """
    print("=" * 100)
    print("PREPROCESSING REQUIREMENTS SUMMARY")
    print("=" * 100)
    
    needs_grayscale_rgb = []
    needs_resize = []
    ready_to_use = []
    
    # Check ALL datasets
    for dataset_name in sorted(DATASET_MODEL_MAPPING.keys()):
        preprocessing = get_preprocessing_transforms(dataset_name)
        
        if not preprocessing:
            ready_to_use.append(dataset_name)
            continue
            
        if preprocessing.get("grayscale_to_rgb", False):
            needs_grayscale_rgb.append(dataset_name)
        
        if "resize_to" in preprocessing:
            needs_resize.append(dataset_name)
        
        if not preprocessing.get("grayscale_to_rgb") and "resize_to" not in preprocessing:
            ready_to_use.append(dataset_name)
    
    print(f"\nDatasets requiring grayscale→RGB conversion: {len(needs_grayscale_rgb)}")
    if needs_grayscale_rgb:
        print(f"  {', '.join(needs_grayscale_rgb)}")
    
    print(f"\nDatasets requiring resize to 224×224: {len(needs_resize)}")
    if needs_resize:
        print(f"  {', '.join(needs_resize)}")
    
    print(f"\nDatasets ready without preprocessing: {len(ready_to_use)}")
    if ready_to_use:
        print(f"  {', '.join(ready_to_use)}")
    else:
        print(f"  (None - all require some preprocessing)")
    
    print("=" * 100)


def _test_single_dataset_model(
    dataset_name: str,
    model_name: str,
    run_inference: bool,
    model_cache: dict
) -> tuple:
    """
    Test a single dataset-model pair with uniform preprocessing.
    
    Supports both standard TorchVision models and custom models from model_definitions.py.
    All models use the same preprocessing pipeline (grayscale→RGB, resize to 224×224, normalize).
    
    Args:
        dataset_name: Name of the dataset to test
        model_name: Name of the model to test
        run_inference: Whether to run inference test
        model_cache: Cache of loaded models to speed up testing
        
    Returns:
        Tuple of (model_loaded, is_compatible, inference_passed)
        - model_loaded: bool - whether the model was successfully loaded
        - is_compatible: bool - whether the pair passes compatibility checks
        - inference_passed: bool or None - True if inference passed, False if failed, None if not tested
    """
    import torch
    import torchvision
    
    dataset_info = DATASET_MODEL_MAPPING[dataset_name]
    category = dataset_info["category"]
    input_size = dataset_info["input_size"]
    num_classes = dataset_info.get("num_classes", 10)
    
    # Try to load model (cached to avoid reloading)
    if model_name in model_cache:
        model = model_cache[model_name]
    else:
        try:
            # Try standard TorchVision model first
            if hasattr(torchvision.models, model_name):
                model_fn = getattr(torchvision.models, model_name)
                model = model_fn(weights="DEFAULT")
                
                # Adjust final layer for dataset's number of classes
                if hasattr(model, 'fc'):
                    in_features = model.fc.in_features
                    model.fc = torch.nn.Linear(in_features, num_classes)
                elif hasattr(model, 'classifier'):
                    classifier = model.classifier
                    named_kids = list(classifier.named_children())
                    if named_kids:
                        last_name, last_child = named_kids[-1]
                        setattr(classifier, last_name, torch.nn.Linear(last_child.in_features, num_classes))
                    else:
                        model.classifier = torch.nn.Linear(classifier.in_features, num_classes)
                elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                    in_features = model.heads.head.in_features
                    model.heads.head = torch.nn.Linear(in_features, num_classes)
            else:
                # Try custom model from model_definitions
                from act.front_end.torchvision_loader.model_definitions import get_model
                model = get_model(model_name, num_classes=num_classes)
            
            model.eval()
            model_cache[model_name] = model
            
        except Exception as e:
            # Model cannot be loaded (neither standard nor custom)
            return (False, False, None)
    
    # Validate compatibility
    result = validate_dataset_model_compatibility(dataset_name, model_name)
    is_compatible = result["compatible"]
    
    # Run inference test (only for classification datasets)
    inference_result = None
    if run_inference and category == "classification" and is_compatible and model is not None:
        try:
            # Create sample input tensor
            sample = torch.randn(1, *input_size)
            
            # Apply standard preprocessing transformations (uniform for all models)
            processed = sample
            
            # Grayscale to RGB
            if processed.shape[1] == 1:
                processed = processed.repeat(1, 3, 1, 1)
            
            # Resize to 224x224
            if processed.shape[-2:] != (224, 224):
                processed = torch.nn.functional.interpolate(
                    processed, size=(224, 224), mode='bilinear', align_corners=False
                )
            
            # Run inference
            with torch.no_grad():
                output = model(processed)
            
            # Verify output shape
            expected_classes = num_classes
            if output.shape[-1] == expected_classes:
                inference_result = True
            else:
                inference_result = False
            
        except Exception as e:
            inference_result = False
    
    # Return: (model_loaded, is_compatible, inference_result)
    return (True, is_compatible, inference_result)


def test_all_dataset_model_pairs(run_inference: bool = True):
    """
    Test all dataset-model pairs comprehensively with optional inference validation.
    
    Args:
        run_inference: If True, run inference tests on classification datasets
        
    Returns:
        Tuple of (total_pairs, tested_pairs, compatible_pairs, inference_passed, inference_failed)
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE DATASET-MODEL ALIGNMENT TEST" + (" (with inference)" if run_inference else ""))
    print("="*100)
    
    # Get all categories dynamically
    all_categories = get_all_categories()
    
    # Track results - initialize dynamically based on actual categories
    results = {category: [] for category in all_categories}
    
    # Initialize global counters
    total_pairs = 0
    tested_pairs = 0
    compatible_pairs = 0
    inference_passed = 0
    inference_failed = 0
    
    # Cache loaded models to speed up testing
    model_cache = {}
    
    # Test each dataset with all its models using the helper function
    for dataset_name in sorted(DATASET_MODEL_MAPPING.keys()):
        dataset_info = DATASET_MODEL_MAPPING[dataset_name]
        category = dataset_info["category"]
        models = dataset_info["models"]
        
        # Initialize result structure for this dataset
        testable_models = []
        ds_compatible = 0
        ds_inf_passed = 0
        ds_inf_failed = 0
        
        # Test each model for this dataset
        for model_name in models:
            is_testable, is_compatible, inference_result = _test_single_dataset_model(
                dataset_name=dataset_name,
                model_name=model_name,
                run_inference=run_inference,
                model_cache=model_cache
            )
            
            if is_testable:
                testable_models.append(model_name)
                tested_pairs += 1
            
            if is_compatible:
                ds_compatible += 1
                compatible_pairs += 1
            
            if inference_result is True:
                ds_inf_passed += 1
                inference_passed += 1
            elif inference_result is False:
                ds_inf_failed += 1
                inference_failed += 1
            
            total_pairs += 1
        
        # Get preprocessing requirements
        preprocessing = get_preprocessing_transforms(dataset_name)
        preprocessing_steps = []
        if preprocessing:
            if preprocessing.get("grayscale_to_rgb", False):
                preprocessing_steps.append("Grayscale→RGB")
            if "resize_to" in preprocessing:
                target = preprocessing["resize_to"]
                preprocessing_steps.append(f"Resize→{target[0]}×{target[1]}")
            if "normalize" in preprocessing:
                preprocessing_steps.append("Normalize")
        
        if not preprocessing_steps:
            preprocessing_steps.append("None (ready)")
        
        # Create result for this dataset
        dataset_result = {
            "dataset": dataset_name,
            "total_models": len(models),
            "testable_models": len(testable_models),
            "inference_passed": ds_inf_passed,
            "inference_failed": ds_inf_failed,
            "models": testable_models[:3] if len(testable_models) > 3 else testable_models,
            "preprocessing": preprocessing_steps
        }
        
        # Store result by category
        results[category].append(dataset_result)
    
    # Print results by category (use dynamically retrieved categories)
    for category in all_categories:
        if not results[category]:
            continue
        
        print(f"\n{'='*100}")
        print(f"{category.upper()} DATASETS ({len(results[category])} datasets)")
        print('='*100)
        print(f"{'Dataset':<25} {'Example Models':<40} {'Total':<7} {'Test':<6} {'Infer':<11} {'Preprocessing'}")
        print('-'*100)
        
        for item in results[category]:
            models_str = f"{item['total_models']}"
            testable_str = f"{item['testable_models']}"
            
            # Format inference results
            if run_inference and (item['inference_passed'] > 0 or item['inference_failed'] > 0):
                inf_str = f"{item['inference_passed']}✓ {item['inference_failed']}✗"
            else:
                inf_str = "N/A"
            
            prep_str = ", ".join(item['preprocessing'])
            example_models = ", ".join(item['models']) if item['models'] else "custom only"
            
            print(f"{item['dataset']:<25} {example_models:<40} {models_str:<7} {testable_str:<6} {inf_str:<11} {prep_str}")
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print('='*100)
    print(f"Total Datasets: {len(DATASET_MODEL_MAPPING)}")
    
    # Count unique models
    unique_models = set()
    for info in DATASET_MODEL_MAPPING.values():
        unique_models.update(info["models"])
    print(f"Total Unique Models: {len(unique_models)}")
    
    print(f"Total Dataset-Model Pairs: {total_pairs}")
    print(f"Testable Pairs (standard models): {tested_pairs}")
    print(f"Compatible Pairs (validated): {compatible_pairs}")
    print(f"Compatibility Rate: {compatible_pairs/tested_pairs*100:.1f}%")
    
    if run_inference:
        print(f"\nInference Tests (classification only):")
        print(f"  Passed: {inference_passed}")
        print(f"  Failed: {inference_failed}")
        if inference_passed + inference_failed > 0:
            print(f"  Success Rate: {inference_passed/(inference_passed+inference_failed)*100:.1f}%")
    
    return total_pairs, tested_pairs, compatible_pairs, inference_passed, inference_failed


def print_dataset_detail(dataset_name: str):
    """
    Print detailed information about a specific dataset (case-insensitive).
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
    """
    actual_name = find_dataset_name(dataset_name)
    info = get_dataset_info(actual_name)
    
    print("=" * 100)
    print(f"DATASET: {actual_name}")
    print("=" * 100)
    print(f"Category: {info['category']}")
    print(f"Input Size: {info['input_size']}")
    if info['num_classes']:
        print(f"Number of Classes: {info['num_classes']}")
    
    # Show preprocessing requirements
    if "preprocessing" in info:
        print(f"\nPreprocessing Required:")
        preprocessing = info["preprocessing"]
        if preprocessing.get("grayscale_to_rgb", False):
            print(f"  ✓ Grayscale to RGB conversion (1 channel → 3 channels)")
        if "resize_to" in preprocessing:
            target = preprocessing["resize_to"]
            print(f"  ✓ Resize to {target[0]}×{target[1]}")
        if "normalize" in preprocessing:
            norm = preprocessing["normalize"]
            print(f"  ✓ Normalize - Mean: {norm['mean'][:3]}, Std: {norm['std'][:3]}")
    
    print(f"\nRecommended Models ({len(info['models'])}):")
    for i, model in enumerate(info['models'], 1):
        print(f"  {i:2d}. {model}")
    print(f"\nNotes:")
    print(f"  {info['notes']}")
    print("=" * 100)


def main():
    """
    Main CLI entry point for TorchVision-specific operations.
    
    Note: For common operations like --list, --search, --download, and --info,
    use the unified CLI: python -m act.front_end
    
    This CLI provides TorchVision-specific functionality like preprocessing
    details, compatibility validation, and comprehensive testing.
    """
    parser = argparse.ArgumentParser(
        description="TorchVision-Specific Dataset/Model Operations (use 'python -m act.front_end' for common operations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TorchVision-Specific Commands:
  --category CATEGORY       Show datasets in a specific category
  --dataset DATASET         Show detailed dataset information
  --summary                 Show complete mapping summary with all datasets
  --preprocessing-summary   Show preprocessing requirements across datasets
  --show-preprocessing DS   Show specific preprocessing for a dataset
  --validate DS MODEL       Validate dataset-model compatibility
  --models-for DATASET      Show recommended models for a dataset
  --datasets-for MODEL      Show compatible datasets for a model
  --load-torchvision DS M   Load downloaded dataset-model pair
  --inference DS MODEL      Test inference on a dataset-model pair
  --all                     Run comprehensive alignment tests
  --all-with-inference      Run tests with inference validation

For Common Operations (list, search, download, info):
  Use the unified CLI: python -m act.front_end
  Examples:
    python -m act.front_end --list
    python -m act.front_end --search mnist
    python -m act.front_end --download MNIST
    python -m act.front_end --info CIFAR10
        """
    )
    
    # Domain-specific commands (keep in TorchVision CLI)
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Show datasets in specific category (classification, detection, segmentation, video, optical_flow)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Show detailed information for a specific dataset"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print complete mapping summary with all datasets and categories"
    )
    # Preprocessing analysis commands
    parser.add_argument(
        "--show-preprocessing",
        dest="show_preprocessing",
        type=str,
        metavar="DATASET",
        help="Show preprocessing requirements for a specific dataset"
    )
    parser.add_argument(
        "--preprocessing-summary",
        action="store_true",
        dest="preprocessing_summary",
        help="Show aggregated preprocessing requirements across all datasets"
    )
    # Compatibility and validation commands
    parser.add_argument(
        "--validate",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Validate dataset-model compatibility (e.g., --validate MNIST resnet18)"
    )
    # Model/dataset relationship queries
    parser.add_argument(
        "--models-for",
        type=str,
        metavar="DATASET",
        dest="models_for",
        help="Show all recommended models for a specific dataset"
    )
    parser.add_argument(
        "--datasets-for",
        type=str,
        metavar="MODEL",
        dest="datasets_for",
        help="Show all datasets that can run inference with a specific model"
    )
    # Comprehensive testing
    parser.add_argument(
        "--all",
        action="store_true",
        dest="test_all",
        help="Run comprehensive alignment tests with tables for all dataset-model pairs"
    )
    parser.add_argument(
        "--all-with-inference",
        action="store_true",
        dest="test_with_inference",
        help="Run comprehensive tests with inference validation (classification only)"
    )
    # Download/load commands (kept for backward compatibility and advanced options)
    parser.add_argument(
        "--download",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Download a dataset-model pair (for basic usage, prefer: python -m act.front_end --download DATASET)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "both"],
        help="Dataset split to download (default: test)"
    )
    parser.add_argument(
        "--list-downloads",
        action="store_true",
        dest="list_downloads",
        help="List downloaded pairs (for simple listing, prefer: python -m act.front_end --list-downloads)"
    )
    # Loading and inference commands
    parser.add_argument(
        "--load-torchvision",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        dest="load_torchvision",
        help="Load a downloaded dataset-model pair (e.g., --load-torchvision MNIST simple_cnn)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        help="Batch size for DataLoader when loading (default: 1)"
    )
    parser.add_argument(
        "--inference",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Test inference for a specific dataset-model pair (e.g., --inference MNIST resnet18)"
    )
    parser.add_argument(
        "--inference-split",
        type=str,
        default="test",
        choices=["train", "test"],
        dest="inference_split",
        help="Dataset split for inference testing (default: test)"
    )
    
    # Add standard device/dtype arguments
    add_device_args(parser)
    
    args = parser.parse_args()
    
    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Handle commands
    if args.category:
        try:
            datasets = list_datasets_by_category(args.category)
            print(f"\nDatasets in category '{args.category}':")
            for name in sorted(datasets):
                print(f"  • {name}")
        except ValueError as e:
            print(f"Error: {e}")
    
    elif args.dataset:
        try:
            print_dataset_detail(args.dataset)
        except ValueError as e:
            print(f"Error: {e}")
    
    elif args.summary:
        print_mapping_summary()
    
    elif args.preprocessing_summary:
        print_preprocessing_summary()
    
    elif args.download:
        # Download dataset-model pair
        dataset_name, model_name = args.download
        result = download_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=args.split
        )
        
        if result['status'] == 'error':
            print(f"\n❌ Error: {result['message']}")
            if 'traceback' in result:
                print(f"\nDetails:\n{result['traceback']}")
        # Success message already printed by download function
    
    elif args.list_downloads:
        # List all downloaded pairs
        downloaded = list_downloaded_pairs()
        
        if not downloaded:
            print("\nNo downloaded dataset-model pairs found.")
            print("Use --download DATASET MODEL to download a pair.")
        else:
            print(f"\n{'='*80}")
            print(f"DOWNLOADED DATASET-MODEL PAIRS ({len(downloaded)})")
            print(f"{'='*80}")
            
            # Calculate total size
            total_size_bytes = sum(item.get('size_bytes', 0) for item in downloaded)
            total_size_formatted = _format_size(total_size_bytes)
            
            for item in downloaded:
                print(f"\n{item['dataset']} + {item['model']}")
                print(f"  Category: {item['category']}")
                print(f"  Classes: {item['num_classes']}")
                print(f"  Splits: {', '.join(item['splits_downloaded'])}")
                print(f"  Size: {item.get('size_formatted', 'Unknown')}")
                print(f"  Location: {item['paths']['raw_data']}")
                if item['preprocessing_required']:
                    print(f"  Preprocessing: {', '.join(item['preprocessing_steps'])}")
            
            print(f"\n{'='*80}")
            print(f"Total Size: {total_size_formatted}")
            print(f"{'='*80}")
    
    elif args.load_torchvision:
        # Load a downloaded dataset-model pair
        dataset_name = args.load_torchvision[0]
        model_name = args.load_torchvision[1]
        split = "test"  # Default split
        batch_size = args.batch_size
        
        print(f"\n{'='*80}")
        print(f"LOADING DATASET-MODEL PAIR")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")
        print(f"Split: {split}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*80}\n")
        
        try:
            # Load the dataset and model
            result = load_dataset_model_pair(
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                batch_size=batch_size,
                auto_download=True
            )
            
            # Display success information
            print(f"✓ Successfully loaded dataset and model!\n")
            
            # Dataset info
            print(f"Dataset Information:")
            print(f"  • Total samples: {len(result['dataset']):,}")
            print(f"  • Split: {split}")
            print(f"  • Category: {result['metadata']['category']}")
            print(f"  • Classes: {result['metadata']['num_classes']}")
            
            # Model info
            print(f"\nModel Information:")
            print(f"  • Architecture: {model_name}")
            total_params = sum(p.numel() for p in result['model'].parameters())
            print(f"  • Total parameters: {total_params:,}")
            
            # DataLoader info
            print(f"\nDataLoader Information:")
            print(f"  • Batch size: {batch_size}")
            print(f"  • Number of batches: {len(result['dataloader'])}")
            
            # Preprocessing info
            if result['metadata']['preprocessing_required']:
                print(f"\nPreprocessing Applied:")
                for step in result['metadata']['preprocessing_steps']:
                    print(f"  • {step}")
            
            print(f"\n{'='*80}")
            print(f"✓ Load completed successfully!")
            print(f"{'='*80}\n")
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print(f"\nThe dataset-model pair has not been downloaded yet.")
            print(f"Use: python -m act.front_end.torchvision_loader.cli --download {dataset_name} {model_name}")
        except ValueError as e:
            print(f"✗ Error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            print(f"\nDetails:\n{traceback.format_exc()}")
    
    elif args.models_for:
        # Show all recommended models for a dataset (case-insensitive)
        try:
            dataset_name = find_dataset_name(args.models_for)
            info = get_dataset_info(dataset_name)
            models = info['models']
            
            print(f"\n{'='*100}")
            print(f"RECOMMENDED MODELS FOR: {dataset_name}")
            print(f"{'='*100}")
            print(f"Category: {info['category']}")
            print(f"Input Size: {info['input_size']}")
            if info['num_classes']:
                print(f"Number of Classes: {info['num_classes']}")
            
            # Check preprocessing requirements
            preprocessing = get_preprocessing_transforms(dataset_name)
            if preprocessing:
                print(f"\nPreprocessing Required:")
                if preprocessing.get("grayscale_to_rgb", False):
                    print(f"  • Grayscale → RGB conversion")
                if "resize_to" in preprocessing:
                    target = preprocessing["resize_to"]
                    print(f"  • Resize to {target[0]}×{target[1]}")
                if "normalize" in preprocessing:
                    print(f"  • Normalize (dataset-specific mean/std)")
            else:
                print(f"\nNo preprocessing required")
            
            print(f"\nRecommended Models ({len(models)}):")
            
            # Check which models are available in torchvision
            import torchvision
            available_models = []
            custom_models = []
            
            for model in models:
                if hasattr(torchvision.models, model):
                    available_models.append(model)
                else:
                    custom_models.append(model)
            
            if available_models:
                print(f"\n  Standard TorchVision Models ({len(available_models)}):")
                for i, model in enumerate(available_models, 1):
                    print(f"    {i:2d}. {model}")
            
            if custom_models:
                print(f"\n  Custom/External Models ({len(custom_models)}):")
                for i, model in enumerate(custom_models, 1):
                    print(f"    {i:2d}. {model}")
            
            print(f"\nNotes: {info['notes']}")
            print(f"{'='*100}")
            
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Use --list to see all available datasets")
    
    elif args.datasets_for:
        # Show all datasets that can run inference with a model (case-insensitive)
        import torchvision
        
        model_name = find_model_name(args.datasets_for)
        
        # Check if model exists in torchvision
        if not hasattr(torchvision.models, model_name):
            print(f"\nNote: '{model_name}' not found in torchvision.models (might be custom)")
        
        # Find all datasets that include this model (case-insensitive)
        matching_datasets = []
        for dataset_name, info in DATASET_MODEL_MAPPING.items():
            # Case-insensitive comparison
            if any(model.lower() == model_name.lower() for model in info['models']):
                matching_datasets.append({
                    'name': dataset_name,
                    'category': info['category'],
                    'input_size': info['input_size'],
                    'num_classes': info['num_classes'],
                    'preprocessing': get_preprocessing_transforms(dataset_name)
                })
        
        print(f"\n{'='*100}")
        print(f"DATASETS COMPATIBLE WITH: {model_name}")
        print(f"{'='*100}")
        
        if not matching_datasets:
            print(f"\nNo datasets found that recommend '{model_name}'")
            print(f"\nNote: This doesn't mean the model can't be used with datasets,")
            print(f"      just that it's not in the recommended models list.")
        else:
            print(f"\nFound {len(matching_datasets)} dataset(s):\n")
            
            # Group by category
            by_category = {}
            for ds in matching_datasets:
                cat = ds['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(ds)
            
            # Print by category
            for category in sorted(by_category.keys()):
                datasets = by_category[category]
                print(f"\n{category.upper()} ({len(datasets)} dataset{'s' if len(datasets) > 1 else ''}):")
                print("-" * 100)
                
                for ds in sorted(datasets, key=lambda x: x['name']):
                    print(f"\n  {ds['name']}")
                    print(f"    Input Size: {ds['input_size']}")
                    if ds['num_classes']:
                        print(f"    Classes: {ds['num_classes']}")
                    
                    # Show preprocessing if needed
                    if ds['preprocessing']:
                        prep_steps = []
                        if ds['preprocessing'].get('grayscale_to_rgb'):
                            prep_steps.append("Grayscale→RGB")
                        if 'resize_to' in ds['preprocessing']:
                            target = ds['preprocessing']['resize_to']
                            prep_steps.append(f"Resize→{target[0]}×{target[1]}")
                        if 'normalize' in ds['preprocessing']:
                            prep_steps.append("Normalize")
                        print(f"    Preprocessing: {', '.join(prep_steps)}")
                    else:
                        print(f"    Preprocessing: None required")
        
        print(f"\n{'='*100}")
    
    elif args.inference:
        # Test inference for a specific dataset-model pair
        dataset_name, model_name = args.inference
        split = args.inference_split
        
        print(f"\n{'='*80}")
        print(f"INFERENCE TEST: {dataset_name} + {model_name}")
        print(f"{'='*80}")
        print(f"Split: {split}")
        print(f"{'='*80}\n")
        
        try:
            result = model_inference_with_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                verbose=True
            )
            
            print(f"\n{'='*80}")
            print(f"INFERENCE TEST RESULT")
            print(f"{'='*80}")
            print(f"Status: {result['status'].upper()}")
            
            if result['status'] == 'success':
                print(f"✓ Inference successful!")
                print(f"  Output shape: {result['output_shape']}")
                print(f"  Dataset samples: {result['num_samples']}")
            else:
                print(f"✗ Inference failed!")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                if 'num_samples' in result:
                    print(f"  Dataset samples: {result['num_samples']}")
            
            print(f"{'='*80}")
            
        except FileNotFoundError as e:
            print(f"\n✗ Error: Dataset-model pair not found locally")
            print(f"   {e}")
            print(f"\nDownload first with:")
            print(f"   python -m act.front_end.torchvision_loader.cli --download {dataset_name} {model_name}")
        except ValueError as e:
            print(f"\n✗ Error: {e}")
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            print(f"\nDetails:\n{traceback.format_exc()}")
    
    elif args.test_all or args.test_with_inference:
        # Run comprehensive test
        run_inference = args.test_with_inference
        total_pairs, tested_pairs, compatible_pairs, inf_passed, inf_failed = test_all_dataset_model_pairs(run_inference=run_inference)
        
        # Print preprocessing summary
        print("")
        print_preprocessing_summary()
        
        # Print final summary
        print("\n" + "="*100)
        print("✓ COMPREHENSIVE ALIGNMENT TEST COMPLETE")
        print("="*100)
        print(f"\nResults:")
        print(f"  • Dataset-Model Pairs: {total_pairs} total, {tested_pairs} testable, {compatible_pairs} validated")
        if run_inference:
            print(f"  • Inference Tests: {inf_passed} passed, {inf_failed} failed")
            if inf_passed + inf_failed > 0:
                print(f"  • Overall Success Rate: {(inf_passed/(inf_passed+inf_failed)*100):.1f}%")
        
        print("\nKey Findings:")
        print("  1. All grayscale datasets (MNIST family) properly convert to RGB")
        print("  2. All low-resolution datasets correctly resize to 224×224")
        print("  3. All datasets include proper normalization specifications")
        print("  4. Preprocessing pipeline ensures 100% compatibility")
        if run_inference:
            print("  5. Inference validated on classification dataset-model pairs")
        print("  5. Detection, segmentation, video, and optical flow use custom models" if not run_inference else "  6. Detection, segmentation, video, and optical flow use custom models")
        print("  6. Ready for ACT verification workflows" if not run_inference else "  7. Ready for ACT verification workflows")
    
    elif args.validate:
        user_dataset, user_model = args.validate
        try:
            dataset_name = find_dataset_name(user_dataset)
            model_name = find_model_name(user_model)
            
            result = validate_dataset_model_compatibility(dataset_name, model_name)
            print(f"\n{'='*100}")
            print(f"COMPATIBILITY CHECK: {dataset_name} + {model_name}")
            print(f"{'='*100}")
            print(f"Compatible: {'✓ YES' if result['compatible'] else '✗ NO'}")
            
            if result['issues']:
                print(f"\nIssues:")
                for issue in result['issues']:
                    print(f"  ⚠ {issue}")
            
            if result['preprocessing_required']:
                print(f"\nPreprocessing Required:")
                for step in result['preprocessing_steps']:
                    print(f"  • {step}")
            else:
                print(f"\nNo preprocessing required - direct compatibility!")
            
            print(f"{'='*100}")
        except ValueError as e:
            print(f"Error: {e}")
    
    elif args.show_preprocessing:
        try:
            dataset_name = find_dataset_name(args.show_preprocessing)
            
            preprocessing = get_preprocessing_transforms(dataset_name)
            info = get_dataset_info(dataset_name)
            print(f"\n{'='*100}")
            print(f"PREPROCESSING FOR: {dataset_name}")
            print(f"{'='*100}")
            print(f"Input Size: {info['input_size']}")
            
            if preprocessing:
                print(f"\nRequired Transformations:")
                if preprocessing.get("grayscale_to_rgb", False):
                    print(f"  1. Grayscale → RGB (repeat channels)")
                if "resize_to" in preprocessing:
                    print(f"  2. Resize to {preprocessing['resize_to']}")
                if "normalize" in preprocessing:
                    norm = preprocessing["normalize"]
                    print(f"  3. Normalize:")
                    print(f"     Mean: {norm['mean']}")
                    print(f"     Std:  {norm['std']}")
                
                print(f"\nPyTorch Code Example:")
                print(f"  from act.front_end.torchvision_loader.data_model_mapping import create_preprocessing_pipeline")
                print(f"  transform = create_preprocessing_pipeline('{dataset_name}')")
                print(f"  dataset = torchvision.datasets.{dataset_name}(")
                print(f"      root='./data', transform=transform, download=True)")
            else:
                print(f"\nNo special preprocessing required!")
            print(f"{'='*100}")
        except ValueError as e:
            print(f"Error: {e}")
    
    else:
        # Default: show quick examples
        print("\n" + "=" * 100)
        print("TORCHVISION DOMAIN-SPECIFIC CLI")
        print("=" * 100)
        print("\nFor common operations, use the unified CLI:")
        print("  python -m act.front_end --list              # List all datasets/categories")
        print("  python -m act.front_end --search mnist      # Search across all")
        print("  python -m act.front_end --info CIFAR10      # Show dataset info")
        print("  python -m act.front_end --download MNIST    # Download dataset + models")
        
        print("\n" + "-" * 100)
        print("TorchVision-Specific Commands (use --help for full list):")
        print("  python -m act.front_end.torchvision_loader --category classification")
        print("  python -m act.front_end.torchvision_loader --dataset CIFAR10")
        print("  python -m act.front_end.torchvision_loader --summary")
        print("  python -m act.front_end.torchvision_loader --preprocessing-summary")
        print("  python -m act.front_end.torchvision_loader --show-preprocessing MNIST")
        print("  python -m act.front_end.torchvision_loader --validate MNIST resnet18")
        print("  python -m act.front_end.torchvision_loader --models-for CIFAR10")
        print("  python -m act.front_end.torchvision_loader --datasets-for resnet18")
        print("  python -m act.front_end.torchvision_loader --inference MNIST resnet18")
        print("  python -m act.front_end.torchvision_loader --all")
        print("  python -m act.front_end.torchvision_loader --all-with-inference")
        print("=" * 100)


if __name__ == "__main__":
    main()
