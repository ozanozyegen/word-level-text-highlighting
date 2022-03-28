# Setup the folder hierarchy
setup_ignored:
	@mkdir data
	@mkdir data/external
	@mkdir data/processed
	@mkdir data/interim
	@mkdir data/raw
	@mkdir models
	@mkdir notebooks
	@mkdir references
	@mkdir reports
	@mkdir test
	@mkdir src

# Run all
# Configure paramters for different datasets and models
tune_models:
	@python train/tune_models.py
# Generate metrics and figures for best dataset-model pair after tuning
run_reports:
	@python train/run_reports.py

# Standard
# Anaconda
load:
	@conda env create -f env.yml
export:
	@conda env export > env.yml

# Extra
# Clean __pycache___ for unix
clean_unix:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

# Clean __pycache___ for windows
clean_win:
	@python -c "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
	@python -c "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"




