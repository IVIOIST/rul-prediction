"""
Quick Start Script for LSTM RUL Prediction
Author: Tony

This script provides a simple interface to train and evaluate the LSTM model.
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing.append(package)
    
    print("=" * 60)
    
    if missing:
        print("\nMissing packages detected!")
        print("Install them with:")
        print("  pip install -r lstm_model/requirements.txt")
        print("\nOr individually:")
        for package in missing:
            print(f"  pip install {package}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True


def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")
    print("=" * 60)
    
    train_path = Path('data/processed/train_FD001_processed.csv')
    test_path = Path('data/processed/test_FD001_processed.csv')
    
    if train_path.exists():
        print(f"✓ Training data found: {train_path}")
    else:
        print(f"✗ Training data NOT found: {train_path}")
        return False
    
    if test_path.exists():
        print(f"✓ Test data found: {test_path}")
    else:
        print(f"✗ Test data NOT found: {test_path}")
        return False
    
    print("=" * 60)
    print("\n✓ All data files are present!")
    return True


def train_model():
    """Train the LSTM model"""
    print("\n" + "=" * 60)
    print("Starting LSTM Model Training")
    print("=" * 60)
    print("\nThis may take several minutes to hours depending on your hardware.")
    print("Progress will be displayed below...\n")
    
    # Import and run the main training script
    try:
        from lstm_model.main import main
        main()
        return True
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False


def run_inference():
    """Run inference on test data"""
    print("\n" + "=" * 60)
    print("Running Inference on Test Data")
    print("=" * 60)
    
    model_path = Path('lstm_model/checkpoints/best_lstm_model.pt')
    
    if not model_path.exists():
        print(f"\n✗ Model not found at {model_path}")
        print("Please train the model first (option 2)")
        return False
    
    try:
        from lstm_model.inference import main
        main()
        return True
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        return False


def show_menu():
    """Display menu options"""
    print("\n" + "=" * 60)
    print("LSTM RUL Prediction - Quick Start Menu")
    print("Author: Tony")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Check dependencies and data")
    print("  2. Train LSTM model")
    print("  3. Run inference (predict on test data)")
    print("  4. Install dependencies")
    print("  5. Exit")
    print("=" * 60)


def install_dependencies():
    """Install dependencies using pip"""
    print("\n" + "=" * 60)
    print("Installing Dependencies")
    print("=" * 60)
    
    requirements_file = Path('lstm_model/requirements.txt')
    
    if not requirements_file.exists():
        print(f"✗ Requirements file not found: {requirements_file}")
        return False
    
    print(f"\nInstalling packages from {requirements_file}...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ])
        print("\n✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        return False


def main():
    """Main function"""
    while True:
        show_menu()
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            deps_ok = check_dependencies()
            data_ok = check_data()
            
            if deps_ok and data_ok:
                print("\n✓ System is ready for training!")
            else:
                print("\n✗ Please fix the issues above before training.")
        
        elif choice == '2':
            if not check_dependencies():
                print("\nPlease install dependencies first (option 4)")
                continue
            if not check_data():
                print("\nPlease ensure data files are present")
                continue
            
            train_model()
        
        elif choice == '3':
            if not check_dependencies():
                print("\nPlease install dependencies first (option 4)")
                continue
            
            run_inference()
        
        elif choice == '4':
            install_dependencies()
        
        elif choice == '5':
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("\n✗ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LSTM RUL Prediction System")
    print("ENGG2112 Project - Turbofan Engine RUL Prediction")
    print("Author: Tony")
    print("=" * 60)
    
    main()
