PYTHON := .venv/bin/python

TRAIN := data/e4_hr_train_downsampled.csv
TEST := data/e4_hr_test_downsampled.csv
FOLDS := data/train_5fold.txt
REG := data/regularization.txt

SRC := src

PRED_A := predictions_a.txt
WEIGHTS_A := weights_a.txt

PRED_B := predictions_b.txt
WEIGHTS_B := weights_b.txt
BEST_LAMBDA := bestlambda.txt
CV_ERRORS := crossvalidation_errors.txt

PRED_C := predictions_c.txt


.PHONY: help test run-a run-b run-c run-all \
        evaluate-a evaluate-b evaluate-c evaluate \
        clean clean-cache


help:
	@echo "COL774 Assignment 1"
	@echo
	@echo "make test        - syntax-check Python files"
	@echo "make run-a       - run Part (a)"
	@echo "make run-b       - run Part (b)"
	@echo "make run-c       - run Part (c)"
	@echo "make run-all     - run Parts (a), (b), (c)"
	@echo "make evaluate-a  - evaluate Part (a)"
	@echo "make evaluate-b  - evaluate Part (b)"
	@echo "make evaluate-c  - evaluate Part (c)"
	@echo "make evaluate    - evaluate all existing predictions"
	@echo "make clean       - remove generated output files"
	@echo "make clean-cache - remove Python cache files"


test:
	$(PYTHON) -m py_compile $(SRC)/part_a.py $(SRC)/part_b.py $(SRC)/part_c.py $(SRC)/run_experiments.py $(SRC)/evaluate.py


run-a:
	$(PYTHON) $(SRC)/run_experiments.py a


run-b:
	$(PYTHON) $(SRC)/run_experiments.py b


run-c:
	$(PYTHON) $(SRC)/run_experiments.py c


run-all:
	$(PYTHON) $(SRC)/run_experiments.py all


evaluate-a:
	$(PYTHON) $(SRC)/evaluate.py $(TEST) $(PRED_A)


evaluate-b:
	$(PYTHON) $(SRC)/evaluate.py $(TEST) $(PRED_B)


evaluate-c:
	$(PYTHON) $(SRC)/evaluate.py $(TEST) $(PRED_C)


evaluate:
	$(PYTHON) $(SRC)/evaluate.py $(TEST) $(PRED_A) $(PRED_B) $(PRED_C)


clean:
	rm -f $(PRED_A) $(WEIGHTS_A)
	rm -f $(PRED_B) $(WEIGHTS_B)
	rm -f $(BEST_LAMBDA) $(CV_ERRORS)
	rm -f $(PRED_C)


clean-cache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete