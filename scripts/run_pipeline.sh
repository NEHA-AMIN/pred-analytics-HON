#!/bin/bash
# Pipeline execution wrapper script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_msg $YELLOW "⚠️  Warning: Virtual environment not activated"
    print_msg $YELLOW "   Run: source venv/bin/activate"
    exit 1
fi

# Check if run_pipeline.py exists
if [[ ! -f "run_pipeline.py" ]]; then
    print_msg $RED "❌ Error: run_pipeline.py not found"
    print_msg $RED "   Make sure you're in the project root directory"
    exit 1
fi

# Default: run full pipeline
if [[ $# -eq 0 ]]; then
    print_msg $BLUE "🚀 Running full pipeline..."
    python run_pipeline.py --full
else
    # Pass all arguments to the Python script
    python run_pipeline.py "$@"
fi

# Check exit status
if [[ $? -eq 0 ]]; then
    print_msg $GREEN "✅ Pipeline completed successfully!"
else
    print_msg $RED "❌ Pipeline failed!"
    exit 1
fi
