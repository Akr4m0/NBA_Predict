#!/usr/bin/env python
# NBA Game Prediction - Environment Setup Script

import subprocess
import sys
import os
import venv
import platform

def create_virtual_environment(env_name="nba_prediction_env"):
    """Create a virtual environment for the project."""
    print(f"Creating virtual environment: {env_name}")
    venv.create(env_name, with_pip=True)
    return env_name

def get_activation_command(env_name):
    """Return the appropriate activation command based on OS."""
    system = platform.system()
    if system == "Windows":
        return f"{env_name}\\Scripts\\activate"
    else:  # Linux or MacOS
        return f"source {env_name}/bin/activate"

def install_requirements(env_name):
    """Install all required packages for the NBA prediction project."""
    # Define required packages
    core_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "ipykernel"
    ]
    
    nba_packages = [
        "nba_api",
        "basketball-reference-scraper"
    ]
    
    optional_packages = [
        "xgboost",
        "lightgbm",
        "flask",
        "plotly",
        "streamlit"
    ]
    
    # Determine pip path based on virtual environment
    system = platform.system()
    if system == "Windows":
        pip_path = f"{env_name}\\Scripts\\pip"
    else:  # Linux or MacOS
        pip_path = f"{env_name}/bin/pip"
    
    # Install core packages
    print("\nInstalling core packages...")
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
    for package in core_packages:
        print(f"Installing {package}...")
        subprocess.check_call([pip_path, "install", package])
    
    # Install NBA-specific packages
    print("\nInstalling NBA data packages...")
    for package in nba_packages:
        print(f"Installing {package}...")
        subprocess.check_call([pip_path, "install", package])
    
    # Ask user if they want to install optional packages
    install_optional = input("\nDo you want to install optional packages (XGBoost, LightGBM, Flask, Plotly, Streamlit)? (y/n): ")
    if install_optional.lower() == 'y':
        for package in optional_packages:
            print(f"Installing {package}...")
            subprocess.check_call([pip_path, "install", package])
    
    # Setup Jupyter kernel for the virtual environment
    print("\nSetting up Jupyter kernel...")
    subprocess.check_call([pip_path, "install", "ipykernel"])
    subprocess.check_call([
        f"{env_name}/bin/python" if system != "Windows" else f"{env_name}\\Scripts\\python",
        "-m", "ipykernel", "install", "--user", "--name=nba_prediction"
    ])

def setup_project_structure():
    """Create the initial project directory structure."""
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data_collection",
        "src/features",
        "src/models",
        "src/visualization",
        "docs",
        "tests"
    ]
    
    print("\nCreating project directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create initial README file
    with open("README.md", "w") as f:
        f.write("# NBA Game Prediction with Decision Trees\n\n")
        f.write("This project aims to predict NBA game outcomes using decision tree-based models.\n")
        f.write("See project roadmap for detailed timeline and milestones.\n")

def initialize_git():
    """Initialize git repository with initial files."""
    print("\nInitializing Git repository...")
    subprocess.call(["git", "init"])
    
    # Create .gitignore file
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
nba_prediction_env/

# Jupyter Notebooks
.ipynb_checkpoints

# Virtual Environment
venv/
ENV/

# Data files
*.csv
*.xlsx
*.json
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# API keys and secrets
.env
.env.local

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
    """
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Add .gitkeep files to empty directories
    for directory in ["data/raw", "data/processed"]:
        with open(f"{directory}/.gitkeep", "w") as f:
            pass
    
    print("Created .gitignore file")

def main():
    print("=" * 50)
    print("NBA Game Prediction - Development Environment Setup")
    print("=" * 50)
    
    # Create virtual environment
    env_name = create_virtual_environment()
    
    # Install required packages
    install_requirements(env_name)
    
    # Setup project structure
    setup_project_structure()
    
    # Initialize git repository
    initialize_git()
    
    # Show activation instructions
    activation_cmd = get_activation_command(env_name)
    
    print("\n" + "=" * 50)
    print("Setup Complete! Next steps:")
    print("=" * 50)
    print(f"1. Activate the virtual environment with: {activation_cmd}")
    print("2. Start Jupyter with: jupyter notebook")
    print("3. Begin with the notebooks in the 'notebooks' directory")
    print("=" * 50)

if __name__ == "__main__":
    main()