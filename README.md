# CompStatFinal

This project implements and compares different time series segmentation algorithms, focusing on the ClaSP (ClaSP: parameter-free time series segmentation) algorithm and its variants.

## Project Structure

- `Analysis.py`: Main implementation file containing the ClaSP algorithm and its variants
- `requirements.txt`: Project dependencies

## Features

- Implementation of original ClaSP algorithm
- Alternative implementations using different KNN methods:
  - HNSW (Hierarchical Navigable Small World)
  - FJLT (Fast Johnson-Lindenstrauss Transform)
- Performance comparison between different implementations
- Support for both univariate and multivariate time series

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd CompStatFinal
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analysis:
```bash
python Analysis.py
```

## Dependencies

- numpy
- pandas
- scipy
- claspy
- hnswlib
- matplotlib

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 