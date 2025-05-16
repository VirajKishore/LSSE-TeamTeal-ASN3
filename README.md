# LSSE-TeamTeal-ASN3

This project is a data preprocessing and visualization tool developed using Python and Dash. It focuses on analyzing Firefox telemetry data to uncover user behavior patterns through data transformation and clustering techniques.

## Project Structure

```
.
├── Firefox ar.xlsx            # Aggregated Firefox telemetry data
├── Firefox rf.xlsx            # Refined Firefox telemetry data
├── preprocess.py              # Script for data preprocessing and clustering
├── Procfile                   # Deployment configuration for Heroku
├── requirements.txt           # List of Python dependencies
└── README.md                  # Project documentation
```

## Features

- Preprocesses Firefox telemetry datasets for further analysis
- Performs clustering on telemetry data using scikit-learn
- Provides an interactive dashboard built with Dash and Bootstrap components
- Supports deployment to cloud platforms such as Heroku using Gunicorn

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or above installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Application Locally

To start the application locally:

```bash
python preprocess.py
```

This will launch the Dash application in your default browser.

### Deployment

This application is configured for deployment on Heroku. To deploy:

1. Create a new Heroku app:
    ```bash
    heroku create your-app-name
    ```

2. Push the repository:
    ```bash
    git push heroku main
    ```

Heroku will use the `Procfile` and `requirements.txt` to configure the environment.

## Dependencies

The main libraries and frameworks used include:

- dash
- dash-bootstrap-components
- pandas
- numpy
- openpyxl
- scikit-learn
- gunicorn

Refer to `requirements.txt` for the full list.

## Notes

- Ensure both Excel files (`Firefox ar.xlsx` and `Firefox rf.xlsx`) are formatted correctly and located in the project root directory.
- Preprocessing must be rerun if the input datasets change.
- The clustering logic is implemented in `preprocess.py` using scikit-learn's algorithms.


